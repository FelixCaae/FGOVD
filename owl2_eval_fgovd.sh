python detectors_inferences/owl2_inference.py --dataset benchmarks/1_attributes.json --out owl2_eval_fgovd/1_attributes.pkl --n_hardnegatives 5
python evaluation/evaluate_map.py --predictions owl2_eval_fgovd/1_attributes.pkl --ground_truth benchmarks/1_attributes.json --out owl2_eval_fgovd/1_attributes_result.json --n_hardnegatives 5

python detectors_inferences/owl2_inference.py --dataset benchmarks/2_attributes.json --out owl2_eval_fgovd/2_attributes.pkl --n_hardnegatives 5
python evaluation/evaluate_map.py --predictions owl2_eval_fgovd/2_attributes.pkl --ground_truth benchmarks/2_attributes.json --out owl2_eval_fgovd/2_attributes_result.json --n_hardnegatives 5

python detectors_inferences/owl2_inference.py --dataset benchmarks/3_attributes.json --out owl2_eval_fgovd/3_attributes.pkl --n_hardnegatives 5
python evaluation/evaluate_map.py --predictions owl2_eval_fgovd/3_attributes.pkl --ground_truth benchmarks/3_attributes.json --out owl2_eval_fgovd/3_attributes_result.json --n_hardnegatives 5

python detectors_inferences/owl2_inference.py --dataset benchmarks/shuffle_negatives.json --out owl2_eval_fgovd/shuffle_negatives.pkl --n_hardnegatives 5
python evaluation/evaluate_map.py --predictions owl2_eval_fgovd/shuffle_negatives.pkl --ground_truth benchmarks/shuffle_negatives.json --out owl2_eval_fgovd/shuffle_negatives_result.json --n_hardnegatives 5

python detectors_inferences/owl2_inference.py --dataset benchmarks/color.json --out owl2_eval_fgovd/color.pkl --n_hardnegatives 2
python evaluation/evaluate_map.py --predictions owl2_eval_fgovd/color.pkl --ground_truth benchmarks/color.json --out owl2_eval_fgovd/color_result.json --n_hardnegatives 2

python detectors_inferences/owl2_inference.py --dataset benchmarks/material.json --out owl2_eval_fgovd/material.pkl --n_hardnegatives 2
python evaluation/evaluate_map.py --predictions owl2_eval_fgovd/material.pkl --ground_truth benchmarks/material.json --out owl2_eval_fgovd/material_result.json --n_hardnegatives 2

python detectors_inferences/owl2_inference.py --dataset benchmarks/pattern.json --out owl2_eval_fgovd/pattern.pkl --n_hardnegatives 2
python evaluation/evaluate_map.py --predictions owl2_eval_fgovd/pattern.pkl --ground_truth benchmarks/pattern.json --out owl2_eval_fgovd/pattern_result.json --n_hardnegatives 2

python detectors_inferences/owl2_inference.py --dataset benchmarks/transparency.json --out owl2_eval_fgovd/transparency.pkl --n_hardnegatives 2
python evaluation/evaluate_map.py --predictions owl2_eval_fgovd/transparency.pkl --ground_truth benchmarks/transparency.json --out owl2_eval_fgovd/transparency_result.json --n_hardnegatives 2




python detectors_inferences/owl2_inference.py --dataset benchmarks/1_attributes.json --out owl2_eval_fgovd/1_attributes_large.pkl --n_hardnegatives 5 --large
python evaluation/evaluate_map.py --predictions owl2_eval_fgovd/1_attributes_large.pkl --ground_truth benchmarks/1_attributes.json --out owl2_eval_fgovd/1_attributes_large_result.json --n_hardnegatives 5

python detectors_inferences/owl2_inference.py --dataset benchmarks/2_attributes.json --out owl2_eval_fgovd/2_attributes_large.pkl --n_hardnegatives 5 --large
python evaluation/evaluate_map.py --predictions owl2_eval_fgovd/2_attributes_large.pkl --ground_truth benchmarks/2_attributes.json --out owl2_eval_fgovd/2_attributes_large_result.json --n_hardnegatives 5

python detectors_inferences/owl2_inference.py --dataset benchmarks/3_attributes.json --out owl2_eval_fgovd/3_attributes_large.pkl --n_hardnegatives 5 --large
python evaluation/evaluate_map.py --predictions owl2_eval_fgovd/3_attributes_large.pkl --ground_truth benchmarks/3_attributes.json --out owl2_eval_fgovd/3_attributes_large_result.json --n_hardnegatives 5

python detectors_inferences/owl2_inference.py --dataset benchmarks/shuffle_negatives.json --out owl2_eval_fgovd/shuffle_negatives_large.pkl --n_hardnegatives 5 --large
python evaluation/evaluate_map.py --predictions owl2_eval_fgovd/shuffle_negatives_large.pkl --ground_truth benchmarks/shuffle_negatives.json --out owl2_eval_fgovd/shuffle_negatives_large_result.json --n_hardnegatives 5

python detectors_inferences/owl2_inference.py --dataset benchmarks/color.json --out owl2_eval_fgovd/color_large.pkl --n_hardnegatives 2 --large
python evaluation/evaluate_map.py --predictions owl2_eval_fgovd/color_large.pkl --ground_truth benchmarks/color.json --out owl2_eval_fgovd/color_large_result.json --n_hardnegatives 2

python detectors_inferences/owl2_inference.py --dataset benchmarks/material.json --out owl2_eval_fgovd/material_large.pkl --n_hardnegatives 2 --large
python evaluation/evaluate_map.py --predictions owl2_eval_fgovd/material_large.pkl --ground_truth benchmarks/material.json --out owl2_eval_fgovd/material_large_result.json --n_hardnegatives 2

python detectors_inferences/owl2_inference.py --dataset benchmarks/pattern.json --out owl2_eval_fgovd/pattern_large.pkl --n_hardnegatives 2 --large
python evaluation/evaluate_map.py --predictions owl2_eval_fgovd/pattern_large.pkl --ground_truth benchmarks/pattern.json --out owl2_eval_fgovd/pattern_large_result.json --n_hardnegatives 2

python detectors_inferences/owl2_inference.py --dataset benchmarks/transparency.json --out owl2_eval_fgovd/transparency_large.pkl --n_hardnegatives 2 --large
python evaluation/evaluate_map.py --predictions owl2_eval_fgovd/transparency_large.pkl --ground_truth benchmarks/transparency.json --out owl2_eval_fgovd/transparency_large_result.json --n_hardnegatives 2



