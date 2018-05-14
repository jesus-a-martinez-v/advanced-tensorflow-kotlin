import org.tensorflow.DataType
import org.tensorflow.Graph
import org.tensorflow.Output
import org.tensorflow.Tensor
import org.tensorflow.types.UInt8

class GraphBuilder(private val graph: Graph) {
    fun div(x: Output<Float>, y: Output<Float>): Output<Float> {
        return binaryOp("Div", x, y)
    }

    fun <T> sub(x: Output<T>, y: Output<T>): Output<T> {
        return binaryOp("Sub", x, y)
    }

    fun <T> resizeBilinear(images: Output<T>, size: Output<Int>): Output<Float> {
        return binaryOp3("ResizeBilinear", images, size)
    }

    fun <T> expandDims(input: Output<T>, dim: Output<Int>): Output<T> {
        return binaryOp3("ExpandDims", input, dim)
    }

    fun <T, U> cast(value: Output<T>, type: Class<U>): Output<U> {
        val dataType = DataType.fromClass(type)
        return graph
                .opBuilder("Cast", "Cast")
                .addInput(value)
                .setAttr("DstT", dataType)
                .build()
                .output(0)
    }

    fun decodeJpeg(contents: Output<String>, channels: Long): Output<UInt8> {
        return graph
                .opBuilder("DecodeJpeg", "DecodeJpeg")
                .addInput(contents)
                .setAttr("channels", channels)
                .build()
                .output(0)
    }

    fun <T> constant(name: String, value: Any, type: Class<T>): Output<T>? {
        return try {
            val tensor: Tensor<T> = Tensor.create(value, type)

            graph
                    .opBuilder("Const", name)
                    .setAttr("dtype", DataType.fromClass(type))
                    .setAttr("value", tensor)
                    .build()
                    .output(0)
        } catch (e: Exception) {
            null
        }
    }

    fun constant(name: String, value: ByteArray): Output<String>? {
        return constant(name, value, String::class.java)
    }

    fun constant(name: String, value: Int): Output<Int>? {
        return constant(name, value, Int::class.java)
    }

    fun constant(name: String, value: IntArray): Output<Int>? {
        return constant(name, value, Int::class.java)
    }

    fun constant(name: String, value: Float): Output<Float>? {
        return constant(name, value, Float::class.java)
    }

    private fun <T> binaryOp(type: String, firstInput: Output<T>, secondOutput: Output<T>): Output<T> {
        return graph
                .opBuilder(type, type)
                .addInput(firstInput)
                .addInput(secondOutput)
                .build()
                .output(0)
    }

    private fun <T, U, V> binaryOp3(type: String, firstInput: Output<U>, secondInput: Output<V>): Output<T> {
        return graph
                .opBuilder(type, type)
                .addInput(firstInput)
                .addInput(secondInput)
                .build()
                .output(0)
    }
}