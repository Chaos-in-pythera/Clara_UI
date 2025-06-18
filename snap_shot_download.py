from huggingface_hub import snapshot_download

snapshot_download(repo_id='THP2903/Qwen2vl_7b_instruct_medical_multiturn_full', local_dir='/home/truongnn/chaos/code/repo/medical_inferneces/model_hf_cached', token='token' )