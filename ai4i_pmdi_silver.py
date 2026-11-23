
from pyspark import pipelines as dp
from pyspark.sql.functions import col, when, to_timestamp
from pyspark.sql.types import BooleanType
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

#O resultado desta função deve ser persistido e atualizado na camada Prata.
#Criação da tabela 'ai4i_pmdi.silver.ai4i_pmdi_silver'.
@dp.materialized_view(name="ai4i_pmdi.silver.ai4i_pmdi_silver")

#Asseguram que registros corrompidos ou com chaves primárias nulas sejam auditados.
#Auditoria de integridade estrutural e de domínio na Camada Prata.
@dp.expect("auditoria_udi_presente", "udi IS NOT NULL")
@dp.expect("auditoria_tipo_valido", "type IN ('L', 'M', 'H')")
def ai4i_pmdi_silver():
    #Leitura da Tabela Bronze
    #Comando Spark para ler a tabela Delta gerada na Camada Bronze e cria o DataFrame Spark 'df_bronze' pronto para a tipagem.
    df_bronze = spark.read.table("ai4i_pmdi.bronze.ai4i_pmdi_bronze")

    #Conversão de colunas de texto para inteiros, doubles e timestamps.
    #Padroniza os tipos de dados fundamentais (UDI, datas, IDs, sensores) antes do processamento avançado.
    #Cria o DataFrame estagiado 'df_silver_stg' com tipos limpos e corretos.
    df_silver_stg = df_bronze.select(
        col("udi").cast("integer"),
        to_timestamp(col("date"), "d/M/yyyy H:mm").alias("date"),
        col("system").cast("integer"),
        col("control").cast("string"),
        col("product_id").cast("string"),
        col("type").cast("string"),
        col("air_temperature_k").cast("double"),
        col("process_temperature_k").cast("double"),
        col("rotational_speed_rpm").cast("double"),
        col("torque_nm").cast("double"),
        col("tool_wear_min").cast("double")
    )

    #Conversão condicional da coluna 'type' ('L', 'M', 'H') em valores numéricos (0.0, 1.0, 2.0).
    #Cria uma escala numérica temporária para que o MICE utilize a hierarquia física de resistência do material (L, M e H) para permitir que as árvores de decisão façam separações matemáticas coerentes entre os regimes de operação. 
    #Coluna auxiliar: entra como preditora no MICE e é descartada antes do yield.
    df_silver_stg = df_silver_stg.withColumn(
        "type_num",
        when(col("type") == "L", 0.0)
        .when(col("type") == "M", 1.0)
        .when(col("type") == "H", 2.0)
    )

    #Definição das Colunas de Sensores Físicos
    #Lista contendo os nomes das colunas de telemetria dos sensores principais.
    #Centraliza as variáveis físicas que passarão por rastreio de nulos, blindagem de domínio e geração de flags no schema final.
    #Define a lista única 'sensor_cols' para controle dos sensores físicos. 
    sensor_cols = ["air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm"]

    # PONTE ENTRE SPARK E PANDAS (visão geral)
    # O IterativeImputer (MICE) do Scikit-Learn opera sobre dados na memória de um único processo Python; o Spark mantém os dados distribuídos.
    # 1. O Spark cria as partições iniciais a partir do layout dos arquivos da tabela Bronze (sem relação com 'system', ou seja, não junta dados da mesma máquina).
    # 2. repartition("system"): reúne cada máquina inteira na mesma partição. Uma partição pode ter várias máquinas, e as linhas não ficam ordenadas.
    # 3. mapInPandas: corta as partições em lotes por quantidade de linhas. Depois converte cada lote em DataFrame Pandas e entrega os lotes a 'process_partition'.
    # 4. process_partition: reúne os lotes, ordena e roda os cálculos sequenciais e o MICE sobre a partição inteira, e devolve o resultado via yield.
    # Cada partição é processada de forma independente, só com os dados dela.
    def process_partition(pdf_batches):

        #Concatenação dos lotes da partição em um único DataFrame Pandas
        #Consome o iterador de lotes e empilha os lotes com pd.concat. Os lotes já chegam como Pandas; a conversão é feita pelo mapInPandas.
        #Os lotes são fatias consecutivas da partição, cortadas por quantidade de linhas, e as linhas não estão ordenadas por máquina.
        #Um lote pode misturar máquinas e dividir o histórico de uma delas ao meio. Os cálculos sequenciais precisam do histórico contínuo.
        #DataFrame 'pdf_telemetry' com a partição inteira (um por partição, não um global).
        # O concat apenas empilha: a ordenação por máquina e data é feita logo abaixo.

        batches = list(pdf_batches)
        if not batches: 
            return
        pdf_telemetry = pd.concat(batches, ignore_index=True) 
   
        #Ordena o DataFrame Pandas pelas colunas 'system' e 'date' (máquina e tempo)
        #Garante a correta progressão temporal para o cálculo acumulativo do desgaste da ferramenta / tempo de uso
        pdf_telemetry = pdf_telemetry.sort_values(by=["system", "date"]).reset_index(drop=True) 

        #Flags de rastreio de imputação: colunas booleanas (True = valor originalmente nulo) para os 4 sensores e para tool_wear_min.
        #Criadas antes do desgaste e do MICE, para preservar a informação de quem era nulo.
        #Permitem auditoria posterior e registra só os valores originalmente ausentes que foram modificados.
        for c in sensor_cols:
            pdf_telemetry[f"{c}_is_imputed"] = pdf_telemetry[c].isna()
        pdf_telemetry["tool_wear_min_is_imputed"] = pdf_telemetry["tool_wear_min"].isna()

        #Mapeamento dos incrementos de desgaste industrial por qualidade de material
        #Dicionário associando cada variante ('L', 'M', 'H') ao seu acréscimo em minutos (2.0, 3.0, 5.0).
        #Reflete a regra industrial determinística do desgaste da ferramenta descrita no estudo de referência (Matzka, 2020)
        #Define a base de cálculo para o tempo de uso da ferramenta.
        inc_map = {"L": 2.0, "M": 3.0, "H": 5.0}

        #Reconstrução iterativa do desgaste por máquina (system)
        #Agrupa os dados por máquina e recalcula o desgaste acumulado linha a linha.
        #O desgaste é um processo determinístico cumulativo que não deve ser estimado por IA.
        #Valores de 'tool_wear_min' recalculados com precisão.
        for sys_id, group_df in pdf_telemetry.groupby("system"): 
            prev_wear = 0.0 
            prev_type = None 
            for idx in group_df.index: 
                if prev_type is None: 
                    pdf_telemetry.at[idx, "tool_wear_min"] = 0.0 
                    prev_type = pdf_telemetry.at[idx, "type"] 
                else: 
                    increase = inc_map[prev_type]  
                    calculated_wear = prev_wear + increase 
                    pdf_telemetry.at[idx, "tool_wear_min"] = calculated_wear 
                    prev_wear = calculated_wear
                    prev_type = pdf_telemetry.at[idx, "type"] 

        # -------------------------------------------------------------------------
        # ENGENHARIA DE VARIÁVEIS AUXILIARES PARA A IMPUTAÇÃO MICE
        # -------------------------------------------------------------------------

        #Converte a data em dias numéricos (float) desde a primeira medição da partição, criando a coluna auxiliar 'date_num'.
        #dt.total_seconds() dá o intervalo em segundos e a divisão por 86400 converte para dias.
        #Serve para o MICE, que só lê números, capturar a tendência temporal. A coluna é descartada antes do yield.
        pdf_telemetry["date_num"] = (
            pdf_telemetry["date"] - pdf_telemetry["date"].min()
        ).dt.total_seconds() / 86400 

        #Definição explícita do conjunto ampliado de colunas preditoras para o MICE
        #Combina os sensores físicos ('sensor_cols'), a categoria numérica do material ('type_num'), o tempo ('date_num'), a máquina ('system') e o desgaste ('tool_wear_min').
        #Fornece contexto completo diretamente na matriz do MICE 
        # Cria a lista 'feat_cols' contendo todos os preditores e alvos da imputação multivariada.
        feat_cols = sensor_cols + ["type_num", "date_num", "system", "tool_wear_min"]

        # -------------------------------------------------------------------------
        # IMPUTAÇÃO MULTIVARIADA POR MICE (EXTRATREES)
        # -------------------------------------------------------------------------

        #Instanciação do regressor ExtraTrees otimizado para estabilidade e RAM
        #Modelo de ensemble de árvores extremamente randomizadas para regressão.
        #Mapeia não-linearidades complexas com alta velocidade.
        #Objeto 'estimator_et' configurado com limite de profundidade e tamanho de folha, para baixo consumo de memória RAM.
        estimator_et = ExtraTreesRegressor(
            n_estimators=100, 
            max_depth=15, 
            min_samples_leaf=5, #mínimo de 5 linhas por folha: árvores menores (menos RAM) e previsões mais suaves, menos sobreajustadas
            random_state=42, 
            n_jobs=None
        )

        #Instanciação do IterativeImputer (MICE)
        #Imputador multivariado baseado em rodadas de regressão encadeada.
        #Configura até 10 iterações de imputação utilizando a estratégia de média inicial e ordem aleatória.(10 iterações é o default do scikit learn)
        #Objeto 'imputer' pronto para preencher valores ausentes de forma coerente.
        imputer = IterativeImputer(
            estimator=estimator_et,
            max_iter=10,
            initial_strategy='mean',
            imputation_order='random',
            random_state=42 
        )
        #Execução da imputação multivariada (treinamento e preenchimento)
        #Entrega o subconjunto numérico 'pdf_telemetry[feat_cols]' ao 'imputer' configurado e dispara até 10 rodadas do MICE.
        #Aprende as relações estatísticas entre sensores e preditores auxiliares para estimar e substituir os valores ausentes
        #Retorna 'imputed_matrix', um array NumPy com os valores originais mais as estimativas.
        imputed_matrix = imputer.fit_transform(pdf_telemetry[feat_cols])

        #Construção do DataFrame Pandas com os dados imputados
        #O fit_transform devolve um array sem nomes; aqui ele vira DataFrame para acessar os sensores imputados por nome.
        #Converte a matriz imputada em um DataFrame com os nomes de 'feat_cols' como colunas.
        #Permite o acesso estruturado aos valores estimados para posterior validação física.
        #O índice (0..n-1) coincide com o de pdf_telemetry, o que permite usar a mesma máscara nos dois.
        df_imputed = pd.DataFrame(imputed_matrix, columns=feat_cols)

        # -------------------------------------------------------------------------
        # REALIDADE FÍSICA E TERMODINÂMICA
        # -------------------------------------------------------------------------

        #Restrição de valores não-negativos para os sensores físicos
        #Aplica a operação .clip(lower=0.0) e arredondamento nas colunas de sensores imputadas.
        #Impede que o modelo matemático gere valores físicos impossíveis (ex: temperatura ou torque negativos).
        #Sensores limpos e consistentes com a realidade industrial.
        for c in sensor_cols: # percorre cada sensor físico (ar, processo, rotação, torque)
            mask = pdf_telemetry[f"{c}_is_imputed"] 
            if mask.any(): 
                pdf_telemetry.loc[mask, c] = df_imputed.loc[mask, c].clip(lower=0.0).round(1) # Grava só nas linhas imputadas com a estimativa do MICE, com negativos virando 0.0 e uma casa decimal como padrão

        #Aplicação da lei termodinâmica (Processo >= Ar)
        #Garante que a temperatura do processo nunca seja inferior à temperatura do ar ambiente.
        #Respeita os princípios termodinâmicos da usinagem onde o ambiente refrigera o sistema.
        #Ajusta a temperatura do processo utilizando o máximo entre ela e a temperatura do ar, garantindo que a temperatura do processo nunca seja menor que a do ar
        proc_mask = pdf_telemetry["process_temperature_k_is_imputed"] 
        if proc_mask.any(): 
            pdf_telemetry.loc[proc_mask, "process_temperature_k"] = np.maximum( 
                pdf_telemetry.loc[proc_mask, "process_temperature_k"], 
                pdf_telemetry.loc[proc_mask, "air_temperature_k"]
            )


        #Restauração da ordenação primária baseada na UDI
        #Reordena o DataFrame Pandas utilizando a chave original 'udi'.
        #Assegura que a ordem original das amostras seja preservada.
        #DataFrame ordenado por UDI.
        pdf_telemetry = pdf_telemetry.sort_values("udi").reset_index(drop=True)

        #Remoção de colunas temporárias e auxiliares antes do yield
        #Descarta do DataFrame apenas as colunas de suporte transientes ('type_num',  'date_num').
        #Garante que o schema final de saída descarte apenas o que é auxiliar.
        #DataFrame limpo contendo apenas as colunas oficiais exigidas pelo contrato Prata.
        pdf_telemetry = pdf_telemetry.drop(columns=["type_num", "date_num"])

        #Libera as matrizes intermediárias: as estimativas do MICE já foram copiadas para pdf_telemetry
        del df_imputed, imputed_matrix

        #Envio da partição processada para o pipeline Spark
        #Comando yield gerando o DataFrame Pandas processado ao Spark.
        #Devolve os dados particionados de volta para o motor Spark de forma controlada. É a saída da process_partition.
        # O mapInPandas exige um iterador de DataFrames, por isso yield e não return.
        yield pdf_telemetry

    #Schema de saída do mapInPandas, que o Spark exige declarado de antemão (ele não enxerga o que process_partition devolve).
    #Parte das colunas de df_silver_stg sem 'type_num' (descartada no Pandas) e soma as 5 flags booleanas criadas em process_partition.
    silver_schema = df_silver_stg.drop("type_num").schema
    for c in sensor_cols:
        silver_schema = silver_schema.add(f"{c}_is_imputed", BooleanType())
    silver_schema = silver_schema.add("tool_wear_min_is_imputed", BooleanType())

    #Reúne cada máquina inteira na mesma partição (repartition) e aplica process_partition a cada partição, nos workers (mapInPandas).
    #Cada partição inteira precisa caber na RAM do worker. Só monta o plano: a execução ocorre quando a view é materializada.
    return df_silver_stg.repartition("system").mapInPandas(process_partition, schema=silver_schema) 