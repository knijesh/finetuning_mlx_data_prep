# finetuning_mlx_data_prep
Finetuning Dataset Creation compatible with HF and MLX



Steps to Run :

1. Install the packages using ```pip install -r requirements.txt```
2. Create an .env file with following variables :
   *BAM_API_KEY (BAM API KEY)
   *FILEPATH ="/Users/nijesh/Downloads/comms.pdf"
   *GA_API_KEY = (Watsonx.ai API Key)
   *PROJECT_ID ="f5df1150-d93a-4b65-8eb4-b07a1b7ff29e"
3. Run "prep_data_chunkwise.py"  , Chunwise Q and A will be populated in "trainable_records_chunkwise.csv" in HuggingFace format (not for MLX).
4. Run "rag_data_prep.py" , The Output file "trainable_records_ragwise.txt" Is in MLX format and the file to use.
   
    
