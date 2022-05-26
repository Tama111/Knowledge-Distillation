import tensorlow as tf

img_size = (299, 299, 3)
batch_size = 32
path = '''path name'''
data = tf.keras.preprocessing.image_dataset_from_directory(path,
                                                           image_size = img_size[:2],
                                                           batch_size = batch_size)
class_names = data.class_names
num_classes = len(class_names)

# Teacher model
# Teacher model could be any pre-trained models like vgg, resnet etc...
# or any model build from scratch but should be trained first before using
# This teacher model has 23,851,784 total parameters
pt_model = tf.keras.applications.InceptionV3(include_top = True, weights = 'imagenet')
pt_model.layers[-1].activation = None

teacher_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape = img_size),
        tf.keras.layers.experimental.preprocessing.Rescaling(scale = 1./127.5, offset = -1),
        pt_model
    ])


# Student Model
# This student model has 1,204,115 total parameters
student_model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape = img_size),
    tf.keras.layers.experimental.preprocessing.Rescaling(scale = 1./127.5, offset = -1),
    
    tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(),
    
    tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(),
    
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(),
    
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(),
    
    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(),
    
    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(),
    
    tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units = 2048, activation = 'relu'),
    tf.keras.layers.Dense(units = num_classes)
])


# student_model.summary()
# teacher_model.summary()


class KnowledgeDistiller(object):
    def __init__(self, teacher_model, student_model, alpha, temperature):
        
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.alpha = alpha
        self.temperature = temperature
        
        self.student_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        self.teacher_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()
        
    def get_loss(self, teacher_out, student_out, target):
        student_loss = self.student_loss_obj(target, student_out)
        
        teacher_logit = tf.nn.softmax(teacher_out / self.temperature)
        student_logit = tf.nn.softmax(student_out / self.temperature)
        cls_out = tf.argmax(teacher_logit, axis = -1)
        
        teacher_loss = self.teacher_loss_obj(cls_out, student_logit)
        
        total_loss = ((1 - self.alpha) * student_loss) + (self.alpha * teacher_loss)
        return total_loss
        
    @tf.function
    def train_step(self, img, target):
        
        with tf.GradientTape() as tape:
            teacher_out = self.teacher_model(img)
            student_out = self.student_model(img)
            loss = self.get_loss(teacher_out, student_out, target)
            
        grads = tape.gradient(loss, student_out.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, loss))
        
        return loss
    
    def train(self, data, epochs = 1):
        losses = []
        for e in range(epochs):
            print(f'Epoch: {e} Starts')
            loss = 0
            for img, tar in data:
                loss += train_step(img, tar)
                print('.', end = '')
                
            loss /= len(data)
            losses.append(loss)
            print(f'\nLoss: {loss}')
            print(f'Epoch: {e} Ends\n')
            
        return losses


