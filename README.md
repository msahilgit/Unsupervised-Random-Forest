## Unbiased learning of protein conformational representation via unsupervised random forest

![Alt text](urf.png)

Accurate data representation is paramount in molecular dynamics simulations to capture the functionally relevant motions of proteins. Traditional feature selection methods, while effective, often rely on labeled data, limiting their applicability to novel systems. Here, we present unsupervised random forest (URF), a self-supervised adaptation of traditional random forests that identifies functionally critical features without requiring prior labels. URF-selected features highlight key functional regions, enabling the identification of important residues in diverse proteins. By implementing a memory-efficient version, we demonstrate URF's capability to resolve functional states in around 10 diverse systems, including folded and intrinsically disordered proteins, performing on par with or surpassing 16 leading baseline methods. Crucially, URF is guided by an internal metric, the learning coefficient, which automates hyper-parameter optimization, making the method robust and user-friendly. Benchmarking results reveal URF's distinct ability to produce functionally meaningful representations in comparison to previously reported methods, facilitating downstream analyses such as Markov state modeling . The investigation presented here establishes URF as a leading tool for unsupervised representation learning in protein biophysics.

### Reference
this repository is implementation of URF protocol, corresponding to publication(ref.).

**MAIN** <br>
 ├── **URF** : *the unsupervised-random-forest module* <br>
 │ <br>
 ├── **data** : *scripts for data estimation from MD trajectories* <br>
 │     ├── ASH1 <br>
 │     ├── LJ polymer <br>
 │     ├── P450_binding <br>
 │     ├── P450_channel1 <br>
 │     ├── SIC1 <br>
 │     ├── T4L <br>
 │     ├── asyn <br>
 │     ├── mopR <br>
 │     ├── mopR_ensembles <br>
 │     ├── pASH1 <br>
 │     └── pSIC1 <br>
 │ <br>
 ├── **scripts** : *scripts for reproducibility of results* <br>
 │     ├── 0_python_modules <br>
 │     ├── ASH1 <br>
 │     ├── LJ_polymer <br>
 │     ├── P450_binding <br>
 │     ├── P450_channel1 <br>
 │     ├── SIC1 <br>
 │     ├── T4L <br>
 │     ├── asyn <br>
 │     ├── baseline <br>
 │           ├── mopr    <br>
 │           └── t4l <br>
 │     ├── functional_regions <br>
 │           ├── mopr <br>
 │           ├── t4l <br>
 │           └── diffnet <br>
 │     ├── hyperparameters <br>
 │     ├── mopR <br>
 │     ├── msm <br>
 │           ├── mopr <br>
 │           ├── asyn <br>
 │           └── vampnet <br>
 │     ├── optimization <br>
 │     ├── pASH1 <br>
 │     └── pSIC1 <br>
 │ <br>
 └── **usage** : *guidelines/tutorials for using URF*  <br> <br>


### Dependencies 
- Numpy <br>
- scikit-learn <br>
- numba <br>
- copy <br>
- tqdm <br>
- multiprocessing <br>
- sys <br>
- fastcluster <br>
- gc <br>
- pickle <br>
- tables (only for certain functions of proximity_matrix.py, off by default) <br>
- scipy <br>
- joblib <br>

### Installation
```bash
conda create --name urf python=3.9
conda activate urf
git clone https://github.com/msahilgit/Unsupervised-Random-Forest
cd Unsupervised-Random-Forest/
pip install -e .
#also see 'alternative.txt' for use without installation
```

### Usage
```bash
from URF.model import unsupervised_random_forest as urf
dobj=urf()
dobj.fit(data)
lc,fimp=dobj.get_output()
# see usage/t{1,2}.ipynb for details
```


### Quick Links
[![Paper](https://img.shields.io/badge/Paper-darkgreen?style=for-the-badge)](https://github.com/navjeet0211/rf-tica-md)
[![Data](https://img.shields.io/badge/Data-darkred?style=for-the-badge)](https://github.com/navjeet0211/rf-tica-md)
[![RF-MD-TICA](https://img.shields.io/badge/RF--MD--TICA-darkblue?style=for-the-badge)](https://github.com/navjeet0211/rf-tica-md)
[![previous-work](https://img.shields.io/badge/paper-gray?style=for-the-badge)](https://pubs.acs.org/doi/full/10.1021/acs.jctc.2c00932)
