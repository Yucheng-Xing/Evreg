# Evreg

This repository contains the Python implementation of the evidential regression model for time-to-event prediction, proposed in our paper:

"Evidential time-to-event prediction with calibrated uncertainty quantification"

The model computes a degree of belief for the event time occurring within a time interval, without any strict distributional assumption. It quantifies both epistemic and aleatory uncertainties using Gaussian Random Fuzzy Numbers and belief functions, enabling clinicians to make uncertainty-aware survival predictions.

If you find this code useful and use it in your own research, please cite the following papers:

######### Cite our work ########
```bash
@inproceedings{huang2024evidential,
  title={An evidential time-to-event prediction model based on Gaussian random fuzzy numbers},
  author={Huang, Ling and Xing, Yucheng and Denoeux, Thierry and Feng, Mengling},
  booktitle={International Conference on Belief Functions},
  pages={49--57},
  year={2024},
  organization={Springer}
}
@article{huang2025evidential,
  title={Evidential time-to-event prediction with calibrated uncertainty quantification},
  author={Huang, Ling and Xing, Yucheng and Mishra, Swapnil and Den{\oe}ux, Thierry and Feng, Mengling},
  journal={International Journal of Approximate Reasoning},
  pages={109403},
  year={2025},
  publisher={Elsevier}
}
```
