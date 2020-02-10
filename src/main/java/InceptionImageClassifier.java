import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Arrays;

public class InceptionImageClassifier implements AutoCloseable {

    private Graph graph = new Graph();
    private String[] labels;

    InceptionImageClassifier() {

    }


    void load_model(byte[] bytes, String[] labels) throws IOException {
        graph.importGraphDef(bytes);
        this.labels = labels;
    }


    float[] predict_image_prob(BufferedImage image, int imgWidth, int imgHeight, String input_operation, String output_operation) {
        image = ImageUtils.resizeImage(image, imgWidth, imgHeight);

        Tensor<Float> imageTensor = TensorUtils.getImageTensorNormalized(image, imgWidth, imgHeight);

        try (Session sess = new Session(graph);
             Tensor<Float> result =
                     sess.runner().feed(input_operation, imageTensor)
                             .fetch(output_operation).run().get(0).expect(Float.class)) {
            final long[] rshape = result.shape();
            if (result.numDimensions() != 2 || rshape[0] != 1) {
                throw new RuntimeException(
                        String.format(
                                "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                Arrays.toString(rshape)));
            }
            int nlabels = (int) rshape[1];
            return result.copyTo(new float[1][nlabels])[0];
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        return new float[]{-1};
    }

    public String predict_image(BufferedImage image, int imgWidth, int imgHeight, String input_operation, String output_operation) {
        try {
            float[] predicted = predict_image_prob(image, imgWidth, imgHeight, input_operation, output_operation);
            int argmax = 0;
            float max = predicted[0];
            for (int i = 1; i < predicted.length; ++i) {
                if (max < predicted[i]) {
                    max = predicted[i];
                    argmax = i;
                }
            }

            return labels[argmax];

        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return "unknown";
    }

    @Override
    public void close() throws Exception {
        if (graph != null) {
            graph.close();
            graph = null;
        }
    }
}