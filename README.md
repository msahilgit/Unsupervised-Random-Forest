## Unbiased learning of protein conformational representation via unsupervised random forest

![Alt text](urf.png)

Accurate data representation is paramount in molecular dynamics simulations to capture the functionally relevant motions of proteins. Traditional feature selection methods, while effective, often rely on labeled data, limiting their applicability to novel systems. Here, we present unsupervised random forest (URF), a self-supervised adaptation of traditional random forests that identifies functionally critical features without requiring prior labels. URF-selected features highlight key functional regions, enabling the identification of important residues in diverse proteins. By implementing a memory-efficient version, we demonstrate URF's capability to resolve functional states in around 10 diverse systems, including folded and intrinsically disordered proteins, performing on par with or surpassing 16 leading baseline methods. Crucially, URF is guided by an internal metric, the learning coefficient, which automates hyper-parameter optimization, making the method robust and user-friendly. Benchmarking results reveal URF's distinct ability to produce functionally meaningful representations in comparison to previously reported methods, facilitating downstream analyses such as Markov state modeling . The investigation presented here establishes URF as a leading tool for unsupervised representation learning in protein biophysics.


this repository is implementation of URF protocol, corresponding to publication(ref.).

This path constitutes subpaths: 

  URF: the URF module for implementation, includes __versions__ for required libraries and their versions (not strict requirement). 
  
  usage: tutorials for URF implementation 
  
  data: scripts for data estimation from MD trajectories, as used in ref. 
  
  scripts: python scripts for reproducing the results in ref.
  

ref: --to be published


MAIN
 ├── URF
 │
 ├── data 
 │     ├── ASH1
 │     ├── LJ polymer
 │     ├── P450_binding
 │     ├── P450_channel1
 │     ├── SIC1
 │     ├── T4L
 │     ├── asyn
 │     ├── mopR
 │     ├── mopR_ensembles
 │     ├── pASH1
 │     └── pSIC1
 │
 ├── scripts
 │     ├── 0_python_modules
 │     ├── ASH1
 │     ├── LJ_polymer
 │     ├── P450_binding
 │     ├── P450_channel1
 │     ├── SIC1
 │     ├── T4L
 │     ├── asyn
 │     ├── baseline
 │           ├── mopr   
 │           └── t4l
 │     ├── functional_regions
 │           ├── mopr
 │           ├── t4l
 │           └── diffnet
 │     ├── hyperparameters
 │     ├── mopR
 │     ├── msm
 │           ├── mopr
 │           ├── asyn
 │           └── vampnet
 │     ├── optimization
 │     ├── pASH1
 │     └── pSIC1
 │
 └── usage



MAIN <br>
├── f1 <br>
│   ├── f11 <br>
│   └── f12 <br>
└── f2 <br>
    ├── f21 <br>
    └── f22 <br>
