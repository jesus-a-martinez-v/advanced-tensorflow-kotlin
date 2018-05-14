import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.TensorFlow
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.*

fun printUsage() {
    val url = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"

    println("Kotlin program that uses a pre-trained Inception model (http://arxiv.org/abs/1512.00567)")
    println("to label JPEG images.")
    println("TensorFlow version: ${TensorFlow.version()}")
    println()
    println("Usage: label_image <model_dir> <image_file>")
    println()
    println("Where: ")
    println("<model_dir> is a directory containing the unzipped contents of the inception model.")
    println("    (from $url)")
    println("<image_file> is the path to a JPEG image file")
}

fun main(args: Array<String>) {
    if (args.size != 2) {
        printUsage()
        System.exit(1)
    }

    val modelDir = args[0]
    val imageFile = args[1]

    val graphDef = readAllBytesOrExit(Paths.get(modelDir, "tensorflow_inception_graph.pb"))
    val labels = readAllLinesOrExit(Paths.get(modelDir, "imagenet_comp_graph_label_strings.txt"))
    val imageBytes = readAllBytesOrExit(Paths.get(imageFile))

    val image = constructAndExecuteGraphToNormalizeImage(imageBytes!!)
    val labelProbabilities = executeInceptionGraph(graphDef!!, image)
    val bestLabelIndex = maxIndex(labelProbabilities)

    println("BEST MATCH: ${labels!![bestLabelIndex]} (${labelProbabilities[bestLabelIndex] * 100}% likely)")
}

fun constructAndExecuteGraphToNormalizeImage(imageBytes: ByteArray): Tensor<Float> {
    val graph = Graph()
    val graphBuilder = GraphBuilder(graph)

    val height = 224
    val weight = 224
    val mean = 117f
    val scale = 1f

    val input = graphBuilder.constant("input", imageBytes)
    val output = graphBuilder.div(
            graphBuilder.sub(
                    graphBuilder.resizeBilinear(
                            graphBuilder.expandDims(
                                    graphBuilder.cast(graphBuilder.decodeJpeg(input!!, 3), Float::class.java),
                                    graphBuilder.constant("make_batch", 0)!!
                            ),
                            graphBuilder.constant("size", intArrayOf(height, weight))!!
                    ),
                    graphBuilder.constant("mean", mean)!!
            ),
            graphBuilder.constant("scale", scale)!!
    )

    val session = Session(graph)
    return session
            .runner()
            .fetch(output.op().name())
            .run()[0]
            .expect(Float::class.java)
}

fun executeInceptionGraph(graphDef: ByteArray, image: Tensor<Float>): FloatArray {
    val graph = Graph()
    graph.importGraphDef(graphDef)

    val session = Session(graph)
    val result = session
            .runner()
            .feed("input", image)
            .fetch("output")
            .run()[0]
            .expect(Float::class.java)
    val resultShape = result.shape()

    if (result.numDimensions() != 2 || resultShape[0] != 1L) {
        throw RuntimeException("Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape ${Arrays.toString(resultShape)}")
    }

    val numberOfLabels = resultShape[1].toInt()
    return result.copyTo(arrayOf(FloatArray(numberOfLabels)))[0]
}
fun maxIndex(probabilities: FloatArray): Int {
    return probabilities.zip(1..probabilities.size).fold(0, { bestIndex, (probability, index) ->
        if (probability > probabilities[bestIndex])
            index
        else
            bestIndex
    })
}
fun readAllBytesOrExit(path: Path): ByteArray? {
    return try {
        Files.readAllBytes(path)
    } catch (e: IOException) {
        System.err.print("Failed to read [$path]: ${e.message}")
        System.exit(0)
        null
    }
}

fun readAllLinesOrExit(path: Path): List<String>? {
    return try {
        Files.readAllLines(path, Charsets.UTF_8)
    } catch (e: IOException) {
        System.err.println("Failed to read [$path]: ${e.message}")
        null
    }
}

