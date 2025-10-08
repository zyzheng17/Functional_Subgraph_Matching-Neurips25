# Functional Matching of Logic Subgraphs: Beyond Structural Isomorphism
## 1. Environment
* **Conda environment**
    * Details can be found in `./environment.yaml`
* **Data Processing**
    * ABC: Instruction can be found in [berkeley-abc](https://github.com/berkeley-abc/abc);
    * Yosys: We suggest users install yosys throught [Conda-eda](https://hdl.github.io/conda-eda/).

## 2. Data Process
1. **Download Raw data**(.aig flile)
    * ITC99: https://github.com/cad-polito-it/I99T
    * OpenABCD: https://github.com/NYU-MLDA/OpenABC
    * ForgeEDA: https://github.com/cure-lab/LCM-Dataset
2. Modify the **path** in `./script/data/overall.sh`
3. Run `bash ./script/data/overall.sh`
<!-- 4. We also provide processed data in  -->

## 3. Run our model
* **Stage #1** Functional Subgraph Detection
    * `bash ./script/run_stage1.sh`
* **Stage #2** Fuzzy Boundary Identification
    * `bash ./script/run_stage2.sh`