# Unsupervised-Random-Forest for resolving protein conformational free energy landscape

Firstly, supervised versions of Random Forest (RF) is common in machine learning, unsupervised Random Forest (urf) is rarely used and not even implemented in python libraries. Here, we provide unsupervised version of random forest and how it is of critical importance to understanding of protein functions.

In our preceeding work featured in Journal of Chemical Theory and computation (https://doi.org/10.1021/acs.jctc.2c00932), we show that high dimensional conformational space of protein (using time lagged independent analysis (tica) projection) may not necessarily be resolved i.e., the functional states of protein are not exclusively segregated on tica derived free energy surface (FES). Shortlisting of input feature space via RF based feature importance scores resolved the conformational space and can be used to study protein allostery, a pipeline called "RF-TICA-MD". Though useful, the requirement of labels (for supervised RF) restricts usage of RF-MD-TICA protocol to protein systems with pre-conseived understanding of functional states (to make labels).

In our on-going work, we are persuing the URF protocol to resolve protein's conformational landscape and implemented it in python.

This is an on-going work. Complete codes and results shall be out soon .... AN SMALL PRESENTATION (peek_into_results.pdf) IS PROVIDED, IF ANYONE IS INTERESTED. PLEASE FEEL FREE TO CONTACT FOR ANY DISCUSSION VIA LINKEDIN OR TWITTER.




