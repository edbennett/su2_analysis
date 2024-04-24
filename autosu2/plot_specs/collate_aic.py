#!/usr/bin/env python3

import shutil


def generate(data, ensembles):
    shutil.copyfile(
        "processed_data/nf1_ADJ/b2.2/m-1.378/48x24/modenumber_grad_plot.pdf",
        "final_plots/DB4M11_modenumber_grad_plot.pdf",
    )
    shutil.copyfile(
        "processed_data/nf1_ADJ/b2.2/m-1.378/48x24/modenumber_slice_plot.pdf",
        "final_plots/DB4M11_modenumber_slice_plot.pdf",
    )
