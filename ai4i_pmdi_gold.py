from pyspark import pipelines as dp
from pyspark.sql.functions import col, when, abs, rand
from math import pi #Para o cálculo da velocidade angulas em rad/s
@dp.materialized_view(name="ai4i_pmdi.gold.ai4i_pmdi_gold")
def ai4i_pmdi_gold():
    
    # Leitura Ativa da Camada Prata.
    # O QUE É: Ingere a tabela previamente processada na Camada Silver.
    # POR QUE: Traz os dados em seu estado limpo, já tipados corretamente e com as imputações termodinâmicas e mecânicas aplicadas.
    df_silver = spark.read.table("ai4i_pmdi.silver.ai4i_pmdi_silver")
    
    # =======================================================================================
    # ENGENHARIA DE FEATURES E VARIÁVEIS FÍSICAS (Base para as falhas)
    # =======================================================================================
    
    # 6. Cálculo da Diferença de Temperatura (Kelvin).
    # O QUE É: Obtém a magnitude da diferença térmica entre o processo e o ar ambiente.
    # RESULTADO ESPERADO: A coluna 'temp_diff_k' contendo um valor numérico absoluto.
    df_gold = df_silver.withColumn(
        "temp_diff_k", 
        abs(col("process_temperature_k") - col("air_temperature_k"))
    )
    
    # 7. Conversão da Velocidade Angular (de RPM para rad/s).
    # O QUE É: A velocidade informada em rotações por minuto é multiplicada por (2 * Pi / 60).
    # RESULTADO ESPERADO: Coluna 'rotational_speed_rad_s' formatada no Sistema Internacional de Medidas.
    df_gold = df_gold.withColumn(
        "rotational_speed_rad_s", 
        col("rotational_speed_rpm") * (2 * pi / 60)
    )
    
    # 8. Cálculo da Potência de Usinagem (Watts).
    # O QUE É: Calcula a potência exigida pelo processo através do produto entre o Torque e a Velocidade Angular.
    # RESULTADO ESPERADO: Coluna 'power_watts', crucial para engatilhar as falhas de suprimento elétrico.
    df_gold = df_gold.withColumn(
        "power_watts", 
        col("torque_nm") * col("rotational_speed_rad_s")
    )
    
    # 9. Cálculo da Tensão Acumulada (minNm - Minuto-Newton Metro).
    # O QUE É: Mede o impacto mecânico latente através do produto do tempo de desgaste pelo torque.
    # RESULTADO ESPERADO: Coluna 'wear_torque_minnm' para medir a fadiga baseada no tipo do material.
    df_gold = df_gold.withColumn(
        "wear_torque_minnm", 
        col("tool_wear_min") * col("torque_nm")
    )
    
    # =======================================================================================
    # DIAGNÓSTICO INDIVIDUAL DAS 5 FALHAS INDEPENDENTES
    # Três falhas dependem de limites determinísticos (HDF, PWF, OSF) e duas 
    # dependem de limites estocásticos/probabilísticos (TWF, RNF)
    # =======================================================================================
    
    # Falha 1: Tool Wear Failure (TWF) - Desgaste Excessivo
    # REGRA: A chance da ferramenta quebrar aumenta drasticamente quando o desgaste atinge a janela de 200 a 240 minutos.
    # APLICAÇÃO: O limite probabilístico foi definido como 51 falhas em 120 trocas (42.5%) (seguindo resultados de Matzka).
    # Usa-se um "seed" fixo (42) no gerador pseudoaleatório (rand) para garantir a reprodutibilidade do dataset sintético.
    df_gold = df_gold.withColumn(
        "twf",
        when(
            (col("tool_wear_min") >= 200) & 
            (col("tool_wear_min") <= 240) & 
            (rand(seed=42) <= (51.0 / 120.0)), 
            1
        ).otherwise(0)
    )
    
    # Falha 2: Heat Dissipation Failure (HDF) - Falha Térmica
    # REGRA: A dissipação térmica é insuficiente se a diferença de temperatura (temp_diff_k) cair abaixo de 8.6K 
    # e a velocidade rotacional (rpm) estiver abaixo de 1380 RPM.
    # APLICAÇÃO: Operador lógico AND (&) combinando as duas restrições numéricas.
    df_gold = df_gold.withColumn(
        "hdf",
        when(
            (col("temp_diff_k") < 8.6) & 
            (col("rotational_speed_rpm") < 1380), 
            1
        ).otherwise(0)
    )
    
    # Falha 3: Power Failure (PWF) - Falha de Suprimento Elétrico
    # REGRA: A máquina trava se o motor exigir uma potência (power_watts) subdimensionada (< 3500 W) 
    # ou entrar em sobrecarga extrema excedendo o barramento suportado (> 9000 W).
    # APLICAÇÃO: Operador lógico OR (|) encadeando os extremos operacionais.
    df_gold = df_gold.withColumn(
        "pwf",
        when(
            (col("power_watts") < 3500) | 
            (col("power_watts") > 9000), 
            1
        ).otherwise(0)
    )
    
    # Falha 4: Overstrain Failure (OSF) - Ruptura por Sobrecarga de Tração
    # REGRA: O limite de resistência varia de acordo com a pureza do material usinado (Low, Medium, High).
    # APLICAÇÃO: Três condicionais sequenciais (.when) cruzam o tipo (type) com o limiar de fadiga (wear_torque_minnm).
    df_gold = df_gold.withColumn(
        "osf",
        when((col("type") == "L") & (col("wear_torque_minnm") > 11000), 1)
        .when((col("type") == "M") & (col("wear_torque_minnm") > 12000), 1)
        .when((col("type") == "H") & (col("wear_torque_minnm") > 13000), 1)
        .otherwise(0)
    )
    
    # Falha 5: Random Failures (RNF) - Falha de Causa Desconhecida
    # REGRA: Para mimetizar o caos do chão de fábrica, qualquer processo tem 0.1% de chance de abortar espontaneamente.
    # APLICAÇÃO: Simulação via rand() <= 0.001. A semente 99 assegura que os padrões gerados não entrem em 
    # ressonância estatística com a semente 42 da falha TWF.
    df_gold = df_gold.withColumn(
        "rnf",
        when(rand(seed=99) <= 0.001, 1).otherwise(0)
    )
    
    # =======================================================================================
    # DIAGNÓSTICO GLOBAL E LIMPEZA
    # =======================================================================================
    
    # Rótulo de Target (Machine Failure)
    # REGRA: Classificador macro do evento preditivo. Se houver 1 em qualquer uma das cinco colunas (twf, hdf, pwf, osf, rnf),
    # entende-se que o ciclo atual obteve falha
    # APLICAÇÃO: Cadeia lógica OR (|) englobando todas as flags criadas.
    df_gold = df_gold.withColumn(
        "machine_failure",
        when(
            (col("twf") == 1) | 
            (col("hdf") == 1) | 
            (col("pwf") == 1) | 
            (col("osf") == 1) | 
            (col("rnf") == 1), 
            1
        ).otherwise(0)
    )
    
   # Ordenação do Schema e Limpeza (Select)
    # O QUE É: Mapeia explicitamente a ordem final das colunas do DataFrame.
    # POR QUE: Coloca o 'machine_failure' antes das falhas individuais e atua como um filtro, 
    # descartando automaticamente as colunas temporárias e flags de imputação que não foram listadas.
    df_final = df_gold.select(
        "udi",
        "date",
        "system",
        "control",
        "product_id",
        "type",
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "machine_failure", 
        "twf",
        "hdf",
        "pwf",
        "osf",
        "rnf"
    )
    
    # Fechamento e Envio Declarativo
    return df_final