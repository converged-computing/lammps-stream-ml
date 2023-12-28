#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import json

here = os.path.dirname(os.path.abspath(__file__))
results = os.path.join(here, "lammps-predict.json")
with open(results, 'r') as fd:
    results = json.loads(fd.read())

dims = results['dims']
y_pred = results['y_pred']
y_true = results['y_true']

for key in ['dims', 'y_pred', 'y_true']:
    del results[key]
    
colors = ['#e74c3c', '#2ecc71', '#286bc8']
for i, model_name in enumerate(results):
    color = colors[i]
    data = results[model_name]
    pred = y_pred[model_name]
    title = " ".join([x.capitalize() for x in data['model_type'].split('-')])

    # Save file to...
    save_fig = data['model_name']+ '-' + data['model_type'] + '.png'

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(alpha=0.75)
    ax.scatter(pred, y_true, lw=3, color=color, alpha=0.8, label=title, s=2)
    ax.set_ylabel('Actual time (seconds)')
    ax.set_xlabel('Predicted time (seconds)')
    ax.set_title(title)
    fig.savefig(save_fig)
    plt.close()
