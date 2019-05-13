import tensorflow as tf
import tensorflow_probability as tfp
import itertools


class GenerateActorModel:
    def __init__(self, numStateSpace, numActionSpace, learningRateActor):
        self.numStateSpace = numStateSpace
        self.numActionSpace = numActionSpace
        self.learningRateActor = learningRateActor

    def __call__(self, hiddenDepth, hiddenWidth):
        print("Generating Actor model with hidden layers: {} x {} = {}".format(hiddenDepth, hiddenWidth, hiddenDepth*hiddenWidth))
        actorGraph = tf.Graph()
        with actorGraph.as_default():
            with tf.name_scope("inputs"):
                state_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="state_")
                actionLabel_ = tf.placeholder(tf.int32, [None, self.numActionSpace], name="actionLabel_")
                advantages_ = tf.placeholder(tf.float32, [None, ], name="advantages_")

            with tf.name_scope("hidden"):
                initWeight = tf.random_uniform_initializer(-0.03, 0.03)
                initBias = tf.constant_initializer(0.01)
                fullyConnected_ = tf.layers.dense(inputs=state_, units=hiddenWidth, activation=tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias)
                for _ in range(hiddenDepth-1):
                    fullyConnected_ = tf.layers.dense(inputs=fullyConnected_, units=hiddenWidth, activation=tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias)
                allActionActivation_ = tf.layers.dense(inputs = fullyConnected_, units = self.numActionSpace, activation = None, kernel_initializer = initWeight, bias_initializer = initBias)

            with tf.name_scope("outputs"):
                actionDistribution_ = tf.nn.softmax(allActionActivation_, name='actionDistribution_')
                #actionEntropy_ = tf.multiply(tfp.distributions.Categorical(probs=actionDistribution_).entropy(), 1, name='actionEntropy_')
                negLogProb_ = tf.nn.softmax_cross_entropy_with_logits_v2(logits=allActionActivation_,
                                                                         labels=actionLabel_, name='negLogProb_')
                loss_ = tf.reduce_mean(tf.multiply(negLogProb_, advantages_), name='loss_')
                actorLossSummary = tf.summary.scalar("ActorLoss", loss_)

            with tf.name_scope("train"):
                trainOpt_ = tf.train.AdamOptimizer(self.learningRateActor, name='adamOpt_').minimize(loss_)

            actorInit = tf.global_variables_initializer()

            actorSummary = tf.summary.merge_all()

        actorModel = tf.Session(graph=actorGraph)
        actorModel.run(actorInit)

        return actorModel


class GenerateCriticModel:
    def __init__(self, numStateSpace, learningRateCritic):
        self.numStateSpace = numStateSpace
        self.learningRateCritic = learningRateCritic

    def __call__(self, hiddenDepth, hiddenWidth):
        print("Generating Critic model with hidden layers: {} x {} = {}".format(hiddenDepth, hiddenWidth, hiddenDepth*hiddenWidth))
        criticGraph = tf.Graph()
        with criticGraph.as_default():
            with tf.name_scope("inputs"):
                state_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="state_")
                valueTarget_ = tf.placeholder(tf.float32, [None, 1], name="valueTarget_")

            with tf.name_scope("hidden"):
                initWeight = tf.random_uniform_initializer(-0.03, 0.03)
                initBias = tf.constant_initializer(0.001)
                fullyConnected_ = tf.layers.dense(inputs=state_, units=hiddenWidth, activation=tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias)
                for _ in range(hiddenDepth - 1):
                    fullyConnected_ = tf.layers.dense(inputs=fullyConnected_, units=hiddenWidth, activation=tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias)

            with tf.name_scope("outputs"):
                value_ = tf.layers.dense(inputs=fullyConnected_, units=1, activation=None, name='value_', kernel_initializer = initWeight, bias_initializer = initBias)
                diff_ = tf.subtract(valueTarget_, value_, name='diff_')
                loss_ = tf.reduce_mean(tf.square(diff_), name='loss_')
            criticLossSummary = tf.summary.scalar("CriticLoss", loss_)

            with tf.name_scope("train"):
                trainOpt_ = tf.train.AdamOptimizer(self.learningRateCritic, name='adamOpt_').minimize(loss_)

            criticInit = tf.global_variables_initializer()

            criticSummary = tf.summary.merge_all()

        criticModel = tf.Session(graph=criticGraph)
        criticModel.run(criticInit)

        return criticModel


class GeneratePolicyNet:
    def __init__(self, numStateSpace, numActionSpace, learningRate):
        self.numStateSpace = numStateSpace
        self.numActionSpace = numActionSpace
        self.learningRate = learningRate

    def __call__(self, hiddenDepth, hiddenWidth):
        print("Generating Policy Net with hidden layers: {} x {} = {}".format(hiddenDepth, hiddenWidth, hiddenDepth * hiddenWidth))
        with tf.name_scope("inputs"):
            state_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="state_")
            actionLabel_ = tf.placeholder(tf.int32, [None, self.numActionSpace], name="actionLabel_")
            accumulatedRewards_ = tf.placeholder(tf.float32, [None, ], name="accumulatedRewards_")

        with tf.name_scope("hidden"):
            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.001)
            fullyConnected_ = tf.layers.dense(inputs=state_, units=hiddenWidth, activation=tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias)
            for _ in range(hiddenDepth-1):
                fullyConnected_ = tf.layers.dense(inputs=fullyConnected_, units=self.numActionSpace, activation=tf.nn.relu, kernel_initializer = initWeight, bias_initializer = initBias)
            allActionActivation_ = tf.layers.dense(inputs=fullyConnected_, units=self.numActionSpace, activation=None, kernel_initializer = initWeight, bias_initializer = initBias)

        with tf.name_scope("outputs"):
            actionDistribution_ = tf.nn.softmax(allActionActivation_, name='actionDistribution_')
            negLogProb_ = tf.nn.softmax_cross_entropy_with_logits_v2(logits=allActionActivation_, labels=actionLabel_,
                                                                     name='negLogProb_')
            loss_ = tf.reduce_sum(tf.multiply(negLogProb_, accumulatedRewards_), name='loss_')
        tf.summary.scalar("Loss", loss_)

        with tf.name_scope("train"):
            trainOpt_ = tf.train.AdamOptimizer(self.learningRate, name='adamOpt_').minimize(loss_)

        mergedSummary = tf.summary.merge_all()

        model = tf.Session()
        model.run(tf.global_variables_initializer())

        return model


def generateModelDictByDepthAndWidth(generateModel, hiddenDepths, hiddenWidths):
    modelDict = {(w*d, d): generateModel(d, w) for d, w in itertools.product(hiddenDepths, hiddenWidths)}
    print("Model Dict contains: {}".format(modelDict.keys()))
    return modelDict


def generateModelDictByNeuronNumberAndDepth(generateModel, hiddenNeuronNumbers, hiddenDepths):
    modelDict = {(n, d): generateModel(d, round(n/d)) for n, d in itertools.product(hiddenNeuronNumbers, hiddenDepths)}
    print("Model Dict contains: {}".format(modelDict.keys()))
    return modelDict
