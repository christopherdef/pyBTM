#!/bin/bash
# run an toy example for BTM

# input and output directory 


# RIGHT NOW IT TAKES 3 arguments: input, output_dir and # of iterations 
output_dir=$2


mkdir -p ${output_dir}

# the input docs for training
doc_pt=$1
#${input_dir}input

iterations=$3

echo "$iterations"
#${input_dir}cleantemporal_text.txt

echo "=============== Index Docs ============="
# docs after indexing
dwid_pt=${output_dir}/doc_wids.txt
# vocabulary file
voca_pt=${output_dir}/voca.txt
python indexDocs.py $doc_pt $dwid_pt $voca_pt

save_step=200

for K in 2 5 10 20 30 40 50 60 70 80 90 100

do    
    #for K in 10 12 14 16 18 20 22 24 26 28 30
    for niter in $iterations
    #200  
    #  12 14 16 

    do 
        date
        # for alpha in 50/$K
        # for beta in 0.005 0.01
        for beta in 0.01
        
        do
            alpha=`echo "scale=3;50/$K"|bc`

            if [ ! -f "${output_dir}/niter${niter}/model_K_${K}_a_${alpha}_b_${beta}/k${K}.pz_d" ]; then
                model_dir=${output_dir}/niter${niter}/model_K_${K}_a_${alpha}_b_${beta}/
                mkdir -p $output_dir/niter${niter}/model_K_${K}_a_${alpha}_b_${beta}

                # beta=0.005
                # niter=10
                

                ## learning parameters p(z) and p(w|z)
                echo "=============== Topic Learning ============="
                W=`wc -l < $voca_pt` # vocabulary size
                make -C ../src
                echo "../src/btm est $K $W $alpha $beta $niter $save_step $dwid_pt $model_dir"
                ../src/btm est $K $W $alpha $beta $niter $save_step $dwid_pt $model_dir
                date

                ## infer p(z|d) for each doc
                echo "================ Infer P(z|d)==============="
                echo "../src/btm inf sum_b $K $dwid_pt $model_dir"
                ../src/btm inf sum_b $K $dwid_pt $model_dir
                date

                ## output top words of each topic
                echo "================ Topic Display ============="
                python topicDisplay.py $model_dir $K $voca_pt
                date

            fi
        done
	date
    done
done
