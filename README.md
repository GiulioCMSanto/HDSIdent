# HDSIdent: Historical Data Segmentation for System Identification

This library can be used to identify intervals suitable for System Identification scanning historical datasets of industrial processess.

**How to reference this work**:

[SANTO, G. C. M. Data Mining Techniques Applied to Historical Data of Industrial Processes 
as a Tool to Find Time Intervals Suitable for System Identification. Masters dissertation 
– Polytechnic School of the University of São Paulo, São Paulo, Brasil, 2020. 
DOI: 10.13140/RG.2.2.13295.46240](https://www.researchgate.net/publication/347511108_Data_Mining_Techniques_Applied_to_Historical_Data_of_Industrial_Processes_as_a_Tool_to_Find_Time_Intervals_Suitable_for_System_Identification?channel=doi&linkId=5fdf5293a6fdccdcb8e856c4&showFulltext=true)

Please see the **References** section to verify the related works.

### Licensing
This library is licensed under the [MIT license](https://github.com/GiulioCMSanto/HDSIdent/blob/master/LICENSE).

### How to install the library
```pip3 install HDSIdent```

### Code Examples
One can find code examples in the [**notebooks/**](https://github.com/GiulioCMSanto/HDSIdent/tree/master/notebooks) folder. 

There are two main ways of reproducing the provided notebooks:

1) All the notebooks were created in Google's Colaboratory. Therefore, you can just open the desired notebook in GitHub and click on "Open in Colab".

2) You can download the notebook and run it locally (don't forget to install the library through ```pip install HDSIdent```)

### Related Libraries
**HDSIdent** uses the following open-source libraries:

- [scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/master/COPYING)
- [scipy](https://github.com/scipy/scipy/blob/master/LICENSE.txt)
- [pandas](https://github.com/pandas-dev/pandas/blob/master/LICENSE)
- [numpy](https://github.com/numpy/numpy/blob/master/LICENSE.txt)
- [matplotlib](https://github.com/matplotlib/matplotlib/blob/master/LICENSE/LICENSE)
- [seaborn](https://github.com/mwaskom/seaborn/blob/master/LICENSE)
- [joblib](https://github.com/joblib/joblib/blob/master/LICENSE.txt)
- [sympy](https://github.com/sympy/sympy/blob/master/LICENSE)

### References

HDSIdent implements and unifies the methods proposed in the following works:

```
PETTITT, A.N., 1979. A non-parametric approach to the
change-point problem. Appl. Stat. 28, 126–135.

PERETZKI, D. et al. Data mining of historic data for process identification.
In: Proceedings of the 2011 AIChE Annual Meeting, p. 1027–1033, 2011.

SHARDT, Y. A. W.; SHAH, S. L. Segmentation Methods for Model Identification from
Historical Process Data. In: Proceedings of the 19th World Congress.
Cape Town, South Africa: IFAC, 2014. p. 2836–2841.

BITTENCOURT, A. C. et al. An algorithm for finding process identification
intervals from normal operating data. Processes, v. 3, p. 357–383, 2015.

RIBEIRO, A. H.; AGUIRRE, L. A. Selecting transients automatically
for the identification of models for an oil well. IFAC-PapersOnLine,
v. 48, n. 6, p. 154–158, 2015.

PATEL, A. Data Mining of Process Data in Mutlivariable Systems.
Degree project in electrical engineering — Royal Institute of Technology,
Stockholm, Sweden, 2016.

ARENGAS, D.; KROLL, A. A Search Method for Selecting Informative Data in Predominantly
Stationary Historical Records for Multivariable System Identification.
In: Proceedings of the 21st International Conference on System Theory,
Control and Computing (ICSTCC). Sinaia, Romenia: IEEE, 2017a. p. 100–105.

ARENGAS, D.; KROLL, A. Searching for informative intervals in predominantly stationary
data records to support system identification. In: Proceedings of the XXVI International
Conference on Information, Communication and Automation Technologies (ICAT). Sarajevo,
Bosnia-Herzegovina: IEEE, 2017b.

WANG, J. et al. Searching historical data segments for process
identification in feedback control loops. Computers and Chemical
Engineering, v. 112, n. 6, p. 6–16, 2018.
```

The following works are also considered:

```
FACELI, K. et al. Inteligência Artificial: Uma Abordagem de Aprendizado de
Máquina. Rio de Janeiro, Brasil: LTC, 2017. (In portuguese)
        
AGUIRRE, L. A. Introdução à Identificação de Sistemas:
técnicas lineares e não lineares: teoria e aplicação. 4. ed.
Belo Horizonte, Brasil: Editora UFMG, 2015.

SMITH, S. W. Digital Signal Processing. San Diego, California:
California Technical Publishing, 1999.
```
