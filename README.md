# LRFinder
In this repository I will be implementing my own version of the Learning Rate Finder for Keras. 

### Last Updates
- Included momentum to make loss function smoother
- Number of iterations is automatically inferred as the number of batches (i.e., it will always run over a full epoch)
- Set of learning rates are spaced evenly on a log scale (a geometric progression) using np.geospace
- Automatic stop criteria if current_loss > 10 x lowest_loss

### Next Updates
- Use exponential annealing instead of np.geospace (`start * (end/start) ** pct`)

### License
- MIT
