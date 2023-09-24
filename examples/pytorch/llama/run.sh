cd build
make -j32
cd ..
FT_DEBUG_LEVEL=DEBUG mpirun -np 4 python llama_example.py --ckpt_path './ckpt' --lib_path '/home/n2/junsik/lab/FasterTransformer/build/lib/libth_transformer.so' --sample_input_file 'input_ids.csv' --tokenizer_path './tokenizer/tokenizer.model'
