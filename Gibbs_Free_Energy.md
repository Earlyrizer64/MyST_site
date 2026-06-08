# Machine Learning and Gibbs Free Energy

***

## Background Information

Gibbs Free Energy: Determines if a reaction will occur spontaneously (no external energy needed) or non-spontaneously (external energy needed).  

| Spontaneous Reaction: Iron Oxide (Rust) | Nonspontaneous Reaction: Separating of water (Electrolysis) |
| -- | -- |
| <img src="Images/Gibbs_Free_Energy/Spontaneous_Reaction.jpg" width="400" height="400"> | <img src="Images/Gibbs_Free_Energy/Nonspontaneous_Reaction.png" width="400" height="400"> | 
| Water on iron interacts with oxygen to oxidize the iron and cause iron oxide or rust.  This process requires no external energy which makes it a spontaneous process.  | Current is sent through water which is a form of external energy which here is separating the water into the components of hydrogen and oxygen.  Since energy is required for this reaction, it is considered a non-spontaneous reaction.  |

***

Entropy: The disorder of a system in terms of its energy or a measurement of the energy that is present in a system or process but is not available to do work

Second Law of Thermodynamics States entrtopy will always increase over time

$$
\Delta S = \frac{Q_{rev}}{T}
$$

| $\Delta S =$ Entropy | 
| :-- | 
| $Q_{rev} =$ Reversible Heat |
| $T =$ Absolute Temperature |

***

Enthalpy: The total heat content of a thermodynamic system

$$
H = U + PV
$$

| $H =$ Enthalpy |
| :-- |
| $U =$ Internal Energy |
| $PV =$ Pressure multiplied with Volume |

The Internal Energy $U$ of a system is directly related to the molecular system's temperature

The Flow Energy $PV$ of a system is the amount of energy required to maintain the control volume or space taken

The Internal Energy summed with the Flow Energy give you the Enthalpy of a system

***

Enthalpy of Formation: The energy required to create one mole of a molecule from its base elements

Standard Enthalpy of Formation is this energy measured at standard conditions (Pressure = 1 atmosphere and T = 298.15K)

| Example Standard Enthalpy of Formation | Table of Some Standard Enthalpies of Formation | 
| -- | -- |
| <img src="Images/Gibbs_Free_Energy/Enthalpy_Of_Formation.png" width="400" height="400"> | <img src="Images/Gibbs_Free_Energy/Standard_Enthalpy_Of_Formation.png" width="400" height="400"> |
|  | [Standard Enthalpy of Formation](https://chem.libretexts.org/Bookshelves/General_Chemistry/ChemPRIME_(Moore_et_al.)/03%3A_Using_Chemical_Equations_in_Calculations/3.10%3A_Standard_Enthalpies_of_Formation) | 




***

What is the difference between the Lennard-Jones Potential (LJP) and a Machine Learning Potential (MLP)?  Both are created essentially from viewing data.  

LJP is more physically accurate when viewing reults than ML models.  LJP was created by considering the weak attractive term from van der Waals interactions and the repulsion term based on electron cloud overlap.  LJP has a function form that is derived with sound physics and the parameters of the equation are determined experimentally.  On the other hand MLP are solely based on input data from quantum mechanical data.  

| Properties | Lennard-Jones Potential | Machine Learning Potential |
| -- | -- | -- |
| Number of Parameters | 2 | Many |
| Physical Interpretability | High | Low |
| Data Requirements | Low | High |
| Accuracy with Complex Chemistry | Limited | Can be Extremely High |

Also remember that a Machine Learning Potential is only accurate in what it is trained in, and struggles to extrapolate data accurately.  They aren't any fundamental equations with MLPs, they can be seen as a black box.  

***

In this module, you will be running 3 simulations of water at various temperatures and analyzing the data to better understand the state it is in.  For the simulation, you will be using the Many-body Atomic Cluster Expansion (MACE) which is an MLP.  This simulation will all be ran using Google Colab which offers some GPU usage.  You will then analyze the results of the simulations.  

