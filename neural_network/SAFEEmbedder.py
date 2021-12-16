import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# SAFE TEAM
# distributed under license: GPL 3 License http://www.gnu.org/licenses/

class SAFEEmbedder:

    def __init__(self, model_file):
        self.model_file = model_file
        self.session = None
        self.x_1 = None
        self.adj_1 = None
        self.len_1 = None
        self.emb = None

    def loadmodel(self):
        """
        with tf.io.gfile.GFile(self.model_file, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)

        sess = tf.compat.v1.Session(graph=graph)
        self.session = sess
        """
        sess = tf.compat.v1.Session()
        saver = tf.compat.v1.train.import_meta_graph(self.model_file)
        print("model_path = ", self.model_file)
        saver.restore(sess, '/home/dannehl/projects/SAFE-tf2/data/traindata/out/runs/1639570794/checkpoints/model')
        self.session = sess
        print("Successfully restored model and session.")
        return sess

    def get_tensor(self):
        """self.x_1 = self.session.graph.get_tensor_by_name("import/x_1:0")
        self.len_1 = self.session.graph.get_tensor_by_name("import/lengths_1:0")"""
        self.x_1 = self.session.graph.get_tensor_by_name("x_1:0")
        self.len_1 = self.session.graph.get_tensor_by_name("lengths_1:0")

        #self.emb = tf.nn.l2_normalize(self.session.graph.get_tensor_by_name('import/Embedding1/dense/BiasAdd:0'), axis=1)
        self.emb = tf.nn.l2_normalize(self.session.graph.get_tensor_by_name('Embedding1/dense/BiasAdd:0'), axis=1)

    def embedd(self, nodi_input, lengths_input):
        print("RUNNING SESSION")
        out_embedding= self.session.run(self.emb, feed_dict = {
                                                    self.x_1: nodi_input,
                                                    self.len_1: lengths_input})

        return out_embedding
