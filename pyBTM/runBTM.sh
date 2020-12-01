#!/bin/bash
function get_param() {
    echo `python3 -c "import sys, json; print(json.load(open('btm-params'))['$1'])"`
}

K=$(get_param K)
alpha=$(get_param alpha)
beta=$(get_param beta)
niter=$(get_param niter)
home_dir=$(get_param home)
input_fname=$(get_param input)
save_step=501

echo "************ K $K , alpha $alpha , beta $beta , niter $niter"
src_dir=${home_dir}BTM/src/
input_dir=${home_dir}data/
output_dir=${home_dir}btm_output/

model_dir=${output_dir}model/
mkdir -p $output_dir/model 

# the input docs for training
doc_pt=${input_dir}${input_fname}

echo "=============== Index Docs ============="
# docs after indexing
dwid_pt=${output_dir}doc_wids.txt

# vocabulary file
voca_pt=${output_dir}voca.txt

python ${home_dir}indexDocs.py $doc_pt $dwid_pt $voca_pt

## learning parameters p(z) and p(w|z)
echo "=============== Topic Learning ============="
W=`wc -l < $voca_pt` # vocabulary size

make -C ${src_dir}
echo "${src_dir}btm est $K $W $alpha $beta $niter $save_step $dwid_pt $model_dir"
${src_dir}btm est $K $W $alpha $beta $niter $save_step $dwid_pt $model_dir

## infer p(z|d) for each doc
echo "================ Infer P(z|d)==============="
echo "${src_dir}btm inf sum_b $K $dwid_pt $model_dir"
${src_dir}btm inf sum_b $K $dwid_pt $model_dir

## output top words of each topic
echo "================ Topic Display ============="
python ${home_dir}topicDisplay.py $model_dir $K $voca_pt
