import org.tensorflow.TensorFlow;

import javax.imageio.ImageIO;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * A simple, easy to use classifier that uses tensorflow graph
 */
public class EZClassifier extends InceptionImageClassifier {
    private int imgWidth;
    private int imgHeight;
    private String input_operation;
    private String output_operation;

    public EZClassifier(File pb_tf_file, File labels, int imgWidth, int imgHeight, String input_operation, String output_operation) throws Exception {
        this.imgWidth = imgWidth;
        this.imgHeight = imgHeight;
        this.input_operation = input_operation;
        this.output_operation = output_operation;

        if (pb_tf_file.exists() && labels.exists()) {
            try {
                load_model(Files.readAllBytes(pb_tf_file.toPath()), readLabels(labels));
            } catch (Exception exp) {
                System.out.println("Error classifier file");
                exp.printStackTrace();
            }
        } else {
            throw new Exception("File not found");
        }
    }

    /**
     * @param pb_tf_file model file
     * @param labels     labels file
     * @throws Exception file not found
     */
    EZClassifier(File pb_tf_file, File labels) throws Exception {
        super();
        this.imgWidth = 224;
        this.imgHeight = 224;
        this.input_operation = "sequential_1_input";
        this.output_operation = "sequential_3/dense_Dense2/Softmax";

        if (pb_tf_file.exists() && labels.exists()) {
            try {
                load_model(Files.readAllBytes(pb_tf_file.toPath()), readLabels(labels));
            } catch (Exception exp) {
                System.out.println("Error classifier file");
                exp.printStackTrace();
            }
        } else {
            throw new Exception("File not found");
        }
    }


    /**
     * Demo
     *
     * @throws Exception error
     */
    public static void main(String[] args) throws Exception {
        System.out.println(TensorFlow.version());
        EZClassifier ezClassifier = new EZClassifier(new File("example_files\\sample_model.pb"), new File("example_files\\labels.txt"));
        ezClassifier.print_pred(new File("example_files\\me.jpg"));
        ezClassifier.print_pred(new File("example_files\\no_me.jpg"));

    }

    /**
     * Reads labels from file
     *
     * @param labelsf file that contains labels - needs to be formatted so 1 label per line.
     * @return String[] of labels
     * @throws IOException Exception is raised if there is a problem reading from the labels file
     */
    private String[] readLabels(File labelsf) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new FileReader(labelsf));
        String s;
        ArrayList<String> labelsAL = new ArrayList<>();
        while ((s = bufferedReader.readLine()) != null) {
            labelsAL.add(s);
        }
        labelsAL.trimToSize();
        return labelsAL.toArray(new String[0]);
    }


    /**
     * predicts an image based on the neural network
     *
     * @param image image to process
     * @return prediction
     * @throws IOException model failed to process the image file properly
     */
    String predict_image(File image) throws IOException {
        return super.predict_image(ImageIO.read(image), this.imgWidth, this.imgHeight, this.input_operation, this.output_operation);
    }


    /**
     * predicts an image probabilities based on the neural network
     *
     * @param image image to process
     * @return prediction probabilities
     * @throws IOException model failed to process the image file properly
     */
    float[] predict_image_prob(File image) throws IOException {
        return super.predict_image_prob(ImageIO.read(image), this.imgWidth, this.imgHeight, this.input_operation, this.output_operation);
    }

    /**
     * A function that prints both probabilities and results from model
     *
     * @param image image to run
     */
    void print_pred(File image) {
        try {
            System.out.println(Arrays.toString(predict_image_prob(image)));
            System.out.println(predict_image(image));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
