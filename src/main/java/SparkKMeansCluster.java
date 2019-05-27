import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.SQLContext;

import java.util.ArrayList;

/**
 * @Auther:liaohongbing@hisense.com
 * @date:2019/5/15
 * @des
 */
public class SparkKMeansCluster {

    public static void main(String[] args) {
        String dataPath = SparkKMeansCluster.class.getClassLoader().getResource("kmeans_data.txt").getFile();
        System.out.println(dataPath);

        SparkConf conf = new SparkConf().setAppName("KMeansCluster");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        SQLContext sqlContext = new SQLContext(jsc);
        JavaRDD<String> sourceData = jsc.textFile("file:///home/spark-test/kmeans_data.txt");

        JavaRDD<Vector> srcRDD = sourceData.map(new Function<String, Vector>() {
            public Vector call(String line) throws Exception {
                String[] arr = line.split(" ");
                ArrayList<Double> list = new ArrayList<Double>();
                double[] d = new double[3];
                int i = 0;
                for(String num:arr){
                    d[i] = new Double(num);
                    i++;
                }
                return Vectors.dense(d);
            }
        });
        System.out.println("-------------srcRDD"+srcRDD.collect());

        KMeansModel kMeansModel = KMeans.train(srcRDD.rdd(),4,50);

        JavaRDD<String> crossRes = srcRDD.map(new Function<Vector, String>() {
            public String call(Vector v1) throws Exception {
                return v1.toString() + "==>" + kMeansModel.predict(v1);
            }
        });

        System.out.println("---------------crossResRDD"+crossRes.collect());

/*        crossRes.foreach(new VoidFunction<String>() {
            @Override
            public void call(String s) throws Exception {
                System.out.println(s);
            }
        });*/



    }
}
