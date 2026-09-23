"""Recreate the published figure from the accompanying exact-step CSV."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    out=Path(__file__).with_name('results')
    with (out/'loss-ppl-every-100-steps.csv').open() as f:
        rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    tokens=[r['loss_tokens']/1e9 for r in rows]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,(field,title) in zip(axes.flat,[('train_mean_last_100','Training loss: trailing 100-update mean'),('validation_loss','Fixed validation loss'),('validation_ppl','Validation perplexity')]):
        for name,label,color in [('muon','SwiGLU + Muon','#2864b7'),('tiller','SwiGLU + fixed TILLER','#d05c27')]:
            ax.plot(tokens,[r[name+'_'+field] for r in rows],label=label,color=color,lw=1.5)
        ax.set_title(title);ax.set_xlabel('Loss-bearing training tokens (billions)');ax.grid(alpha=.2)
    axes[0,0].legend()
    axes[0,1].set_ylim(2.6,4.5)
    axes[0,1].set_title('Fixed validation loss (zoom: 2.6–4.5)')
    axes[1,0].set_yscale('log')
    axes[1,0].set_ylabel('PPL (log scale)')
    ax=axes[1,1]
    ax.plot(tokens,[r['validation_loss_gap_tiller_minus_muon'] for r in rows],color='#7554a3')
    ax.axhline(0,color='black',lw=.8);ax.set_title('Validation gap: TILLER − Muon (negative favors TILLER)')
    ax.set_xlabel('Loss-bearing training tokens (billions)');ax.grid(alpha=.2)
    fig.suptitle('2.056B parameters · MHA · seed 1337 · 3B-token prefix of a 12B LR schedule')
    fig.savefig(out/'learning-curves.png',dpi=160)
    plt.close(fig)

if __name__=='__main__': main()
