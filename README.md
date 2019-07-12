# LRFinder
Learning Rate Finder Callback for Keras. Proposed by Leslie Smith's at https://arxiv.org/abs/1506.01186. Popularized and encouraged by Jeremy Howard in the [fast.ai deep learning course](https://course.fast.ai/). 

### Last Updates
- Autoreload model's weights for each change of learning rate
- For each learning rate, trains the model over `batches_lr_update` batches
- Compatible with both model.fit_generator and model.fit method
- Allow usage of more than one epoch
- Included momentum to make loss function smoother
- Number of iterations is automatically inferred as the number of batches (i.e., it will always run over a full epoch)
- Set of learning rates are spaced evenly on a log scale (a geometric progression) using np.geospace
- Automatic stop criteria if `current_loss > stop_multiplier * lowest_loss`
- `stop_multiplier` is a linear equation where it is 10 if momentum is 0 and 4 if momentum is 0.9

### Next Updates
- Use exponential annealing instead of np.geospace (`start * (end/start) ** pct`)

### License
- MIT
