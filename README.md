# Functional Matching of Logic Subgraphs: Beyond Structural Isomorphism
## 1. Environment
* **Conda environment**
    * Details can be found in `./environment.yaml`
* **Data Processing**
    * ABC: Instruction can be found in [berkeley-abc](https://github.com/berkeley-abc/abc);
    * Yosys: We suggest users install yosys through [Conda-eda](https://hdl.github.io/conda-eda/).

## 2. Data Process
* **Download Raw Data**(.aig flile)
    * ITC99: https://github.com/cad-polito-it/I99T
    * OpenABCD: https://github.com/NYU-MLDA/OpenABC
    * ForgeEDA: https://github.com/cure-lab/LCM-Dataset and https://huggingface.co/datasets/zshi0616/ForgeEDA_AIG
* **Process Raw Data**
    * Modify the **path** in `./script/data/overall.sh`
    * Run `bash ./script/data/overall.sh`
*  We also provide processed ForgeEDA data in [train_data.pt](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155209933_link_cuhk_edu_hk/EbtwP0z3ew1GmkAc90L1Q0wBWxcCY-ol579kLsc6FcQZzw?e=UOeZwo) and [test_data.pt](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155209933_link_cuhk_edu_hk/Ed8z77JAY29HsCrx1W7oDvQBfV3XSoQT88phrLU9ep7f2A?e=ceeGoz)
<!-- 4. We also provide processed data in  -->

## 3. Run our model
* **Environment Setting** 
    * Modify the dataset path, log path and pretrained chechpoint path in  `./script/run_stage1.sh` and `./script/run_stage1.sh` if needed
* **Stage #1** Functional Subgraph Detection
    * `bash ./script/run_stage1.sh`
* **Stage #2** Fuzzy Boundary Identification
    * `bash ./script/run_stage2.sh`
