from pyspark import pipelines as dp
from pyspark.sql.functions import col

# A função abaixo cria uma tabela física.
#Isso delega ao Databricks a responsabilidade de executar o "overwrite". 
#O Lakeflow agora assume a gravação no disco.
@dp.materialized_view(name="ai4i_pmdi.bronze.ai4i_pmdi_bronze")
def ai4i_pmdi_bronze():
    
    #Define o caminho do arquivo de origem no Volume do Unity Catalog.
    file_path = "/Volumes/ai4i_pmdi/raw/ai4i_pmdi_volume/AI4I-PMDI.csv"

    #Configura e executa a leitura do arquivo CSV.
    raw_df = (
        spark.read.format("csv") 
        #Primeira linha do arquivo tem os nomes das colunas.
        .option("header", "true") 
        # Força tudo a entrar como texto puro para garantir a não corrupção e truncagem de dados
        .option("inferSchema", "false") 
        #Textos com quebra de linha não quebram a tabela.
        .option("multiline", "true") 
        #Define que a vírgula é a divisória matemática das colunas no arquivo original.
        .option("sep", ",") 
        #Executa o plano de leitura e carrega os dados brutos na memória.
        .load(file_path)
    )

    #Lista vazia para armazenar as instruções das colunas limpas.
    transformed_columns = []
    
    #Laço de repetição que lê os nomes originais do CSV.
    for old_col in raw_df.columns:
        
        #Limpeza de metadados:
        #Remove espaços nas pontas (.strip), deixa tudo minúsculo (.lower), e troca espaços e traços por underline (_), além de remover parênteses.
        #Isso evita falhas em querys futuras.
        new_col = old_col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        
        #Constrói a projeção final:
        #Pega a coluna original (old_col), garante que ela seja do tipo texto (.cast("string")),e renomeia para o nome limpo (.alias(new_col)). 
        transformed_columns.append(col(f"`{old_col}`").cast("string").alias(new_col))
    
    #Devolve a tabela processada para o Lakeflow.
    #Não existe '.write' ou '.saveAsTable' aqui. A pipeline que engole esse 'return' persiste os dados na tabela ai4i_pmdi_bronze sem gerar erros. A tabela Bronze é criada e atualizada no disco.
    return raw_df.select(transformed_columns)