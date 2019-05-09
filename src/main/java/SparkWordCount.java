import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

/**
 * @Auther:liaohongbing@hisense.com
 * @date:2019/3/5
 * @des
 */
public class SparkWordCount {
    public static void main(String[] args){
        SparkConf conf = new SparkConf().setAppName("WordCountCluster1");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        System.out.println("appname: "+jsc.appName());
        System.out.println("print args:");
        for(String arg :args){
            System.out.println(arg);
        }

        //读取文件的每一行到RDD
        JavaRDD<String> lines = jsc.textFile("file:///home/spark-test/trainData");
        //对每一行进行拆分，封装成LabeledPoint
        JavaRDD<LabeledPoint> data = lines.map(new Function<String, LabeledPoint>() {
            private static final long serialVersionUID = 1L;
            @Override
            public LabeledPoint call(String str) throws Exception {
                System.out.println(str);
                String[] t1 = str.split(",");
                String[] t2 = t1[1].split(" ");
                LabeledPoint a = new LabeledPoint(Double.parseDouble(t1[0]),
                        Vectors.dense(Double.parseDouble(t2[0]), Double.parseDouble(t2[1]), Double.parseDouble(t2[2])));
                return a;
            }
        });
        //将所有LabeledPoint，随机分成7:3的两个切片
        //小数据做练习时，这一步也可免去，全部作为训练数据得到模型。

/*        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[] { 0.7, 0.3 }, 11L);
        JavaRDD<LabeledPoint> traindata = splits[0];
        JavaRDD<LabeledPoint> testdata = splits[1];*/
        //朴素贝叶斯模型训练
//        final NaiveBayesModel model = NaiveBayes.train(data.rdd());
        final NaiveBayesModel model = NaiveBayesModel.load(jsc.sc(),"file:///home/spark-test/model");
        //测试model，若未切分数据，可免去。
//        model.save(jsc.sc(),"/home/spark-test/model");



        /*JavaPairRDD<Double, Double> predictionAndLabel = testdata
                .mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
                    private static final long serialVersionUID = 1L;
                    @Override
                    public Tuple2<Double, Double> call(LabeledPoint p) {
                        return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
                    }
                });
        //由测试数据得到模型分类精度
        double accuracy = predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
            private static final long serialVersionUID = 1L;
            @Override
            public Boolean call(Tuple2<Double, Double> pl) {
                return pl._1().equals(pl._2());
            }
        }).count() / (double) testdata.count();

        System.out.println("模型精度为："+accuracy);
*/



        //一：直接利用模型计算答案
        JavaRDD<String> testData = jsc.textFile("file:///home/spark-test/testData");


        JavaPairRDD<String, Double> res = testData.mapToPair(new PairFunction<String, String, Double>() {
            private static final long serialVersionUID = 1L;
            @Override
            public Tuple2<String,Double> call(String line) throws Exception{
                String[] t2 = line.split(" ");
                Vector v = Vectors.dense(Double.parseDouble(t2[0]), Double.parseDouble(t2[1]),
                        Double.parseDouble(t2[2]));
                double res = model.predict(v);
                return new Tuple2<String,Double>(line,res);
            }
        });

        res.saveAsTextFile("file:///home/spark-test/result");




        jsc.close();
    }
}
