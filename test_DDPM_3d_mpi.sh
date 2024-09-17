SAMPLE_FLAGS="--batch_size 1 --num_samples 6"
MODEL_FLAGS="--attention_resolutions 1000 --large_size 96 --small_size 96 --num_channels 128 --use_fp16 True --num_head_channels 64 --learn_sigma True --resblock_updown True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"

mpiexec -n 6 python ./scripts/test.py $MODEL_FLAGS --model_path ./checkpoints/model.pt --base_samples sample_PET.npz --save_dir ./results/ $SAMPLE_FLAGS

