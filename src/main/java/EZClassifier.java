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
    /**
     * @param pb_tf_file model file
     * @param labels     labels file
     * @throws Exception file not found
     */
    EZClassifier(File pb_tf_file, File labels) throws Exception {
        super();
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

//    EZClassifier(){
//        FileChooser fileChooser = new FileChooser();
//        fileChooser.setTitle("Open Resource File");
//        fileChooser.showOpenDialog(stage);
//    }

    /**
     * Demo
     *
     * @throws Exception error
     */
    public static void main(String[] args) throws Exception {
        System.out.println(TensorFlow.version());
        EZClassifier ezClassifier = new EZClassifier(new File("example_files\\sample_model.pb"), new File("example_files\\labels.txt"));
        ezClassifier.print_pred_teachable_machine(new File("example_files\\me.jpg"));
        ezClassifier.print_pred_teachable_machine(new File("example_files\\no_me.jpg"));

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
     * predicts an image based on the neural network, assuming a file created by the teachable machine example
     *
     * @param image image to process
     * @return prediction
     * @throws IOException model failed to process the image file properly
     */
    String predict_image_teachable_machine(File image) throws IOException {
        return super.predict_image(ImageIO.read(image), 224, 224, "sequential_1_input", "sequential_3/dense_Dense2/Softmax");
    }

    /**
     * predicts an image probabilities based on the neural network, assuming a file created by the teachable machine example
     *
     * @param image image to process
     * @return prediction probabilities
     * @throws IOException model failed to process the image file properly
     */
    float[] predict_image_prob_teachable_machine(File image) throws IOException {
        return super.predict_image_prob(ImageIO.read(image), 224, 224, "sequential_1_input", "sequential_3/dense_Dense2/Softmax");
    }

    /**
     * A function that prints both probabilities and results from teachable machine model
     *
     * @param image image to run
     */
    void print_pred_teachable_machine(File image) {
        try {
            System.out.println(Arrays.toString(predict_image_prob_teachable_machine(image)));
            System.out.println(predict_image_teachable_machine(image));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
