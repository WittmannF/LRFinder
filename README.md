# LRFinder for Keras
Learning Rate Finder Callback for Keras. Proposed by Leslie Smith's at https://arxiv.org/abs/1506.01186. Popularized and encouraged by Jeremy Howard in the [fast.ai deep learning course](https://course.fast.ai/). 

### Usage
```py
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.datasets import fashion_mnist
!git clone https://github.com/WittmannF/LRFinder.git
from LRFinder.keras_callback import LRFinder

# 1. Input Data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

mean, std = X_train.mean(), X_train.std()
X_train, X_test = (X_train-mean)/std, (X_test-mean)/std

# 2. Define and Compile Model
model = Sequential([Flatten(),
                    Dense(512, activation='relu'),
                    Dense(10, activation='softmax')])

model.compile(loss='sparse_categorical_crossentropy', \
              metrics=['accuracy'], optimizer='sgd')


# 3. Fit using Callback
lr_finder = LRFinder(min_lr=1e-4, max_lr=1)

model.fit(X_train, y_train, batch_size=128, callbacks=[lr_finder], epochs=2)
```

![Screen Shot 2019-07-12 at 17 56 36](https://user-images.githubusercontent.com/5733246/61158150-84382100-a4ce-11e9-9d88-99cd43986b0e.png)

### Last Updates
- Reload weights when training ends
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
- Add a test framework

### License
- MIT
