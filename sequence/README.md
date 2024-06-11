Here, we include code to run O-INR on seqeuntial data such as video, GIF as well as brain-imaging data. Please refer to the "main_seq.py" file for additional arguments. 

Example command to run O-INR on brain imagining data is given below.
```
python main_seq.py	--exp adni \
				--frame_gap 3 \
				--scheduler_en \
				--video_path "path to sequential data" \
				--dataset adni \
				--epoch 200
```
