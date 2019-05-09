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
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;


/**
 * @Auther:liaohongbing@hisense.com
 * @date:2019/5/1
 * @des
 */
public class NaiveBayesAlgorithm {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("SparkNaiveBayesTest");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        for(String arg :args){
            System.out.println("_______________________________________"+arg);
        }
        SQLContext sqlContext = new SQLContext(jsc);

/*        JSONObject params = JSON.parseObject(args[0]);
        JSONObject ds = params.getJSONObject("datasource");
        JSONArray labels = params.getJSONArray("labelColumns");
        JSONArray features = params.getJSONArray("featureColumns");
        String ModelSavePath = params.getString("modelSavePath");*/

        //jdbc.url=jdbc:mysql://localhost:3306/database
//        String url = "jdbc:mysql://"+ds.getString("address")+":"+ds.getString("port")+"/"+ds.getString("database") +"?useUnicode=true&characterEncoding=utf-8&useSSL=false";
        String url = "jdbc:mysql://10.16.4.67:3306/spark_test?useUnicode=true&characterEncoding=utf-8&useSSL=false";
        System.out.println(url);
        //查找的表名
        String table = "bayes_train_data";
        //增加数据库的用户名(user)密码(password),指定test数据库的驱动(driver)
        Properties connectionProperties = new Properties();
        connectionProperties.put("user","root");
        connectionProperties.put("password","root");
        connectionProperties.put("driver","com.mysql.jdbc.Driver");

        //SparkJdbc读取Postgresql的products表内容
//        System.out.println("读取数据库中的bayes_train_data表内容");
        // 读取表中所有数据
        final Dataset<Row> rows = sqlContext.read().jdbc(url,table,connectionProperties).select("*");
        //显示数据
        rows.show();


        JavaRDD<LabeledPoint> trainData = rows.javaRDD().map(new Function<Row, LabeledPoint>() {
            @Override
            public LabeledPoint call(Row row) throws Exception {
                LabeledPoint lp = new LabeledPoint(row.getInt(1), Vectors.dense(row.getInt(2),row.getInt(3),row.getInt(4)));

                System.out.println(lp.label()+"______"+lp.features());
                return lp;
            }
        });

//        trainData.saveAsTextFile("file:///home/spark-test/trainData.txt");

        final NaiveBayesModel model = NaiveBayes.train(trainData.rdd());
//        model.save(jsc.sc(),"file:///home/spark-test/model");

        Dataset<Row> testData = sqlContext.read().jdbc(url,"bayes_test_data",connectionProperties).select("*");

        testData.show();

/*        JavaRDD<LabeledPoint> testDataRDD = testData.javaRDD().map(new Function<Row, LabeledPoint>() {
            @Override
            public LabeledPoint call(Row row) throws Exception {
                LabeledPoint lp = new LabeledPoint(row.getInt(1), Vectors.dense(row.getInt(2),row.getInt(3),row.getInt(4)));
                return lp;
            }
        });*/

        JavaRDD<LabeledPoint> res = testData.javaRDD().map(new Function<Row, LabeledPoint>() {
            @Override
            public LabeledPoint call(Row row) throws Exception {

                Vector v = Vectors.dense(row.getDouble(2),row.getDouble(3),row.getDouble(4));
                double label = model.predict(v);

                return new LabeledPoint(label,v);
            }
        });

//        res.saveAsTextFile("file:///home/spark-test/result115");

        JavaRDD<Row> rowRDD = res.map(new Function<LabeledPoint, Row>() {
            @Override
            public Row call(LabeledPoint labeledPoint) throws Exception {
                return RowFactory.create(labeledPoint.label(),labeledPoint.features().toArray()[0],labeledPoint.features().toArray()[1],labeledPoint.features().toArray()[2]);
            }
        });
//        rowRDD.saveAsTextFile("file:///home/spark-test/rowRDD");

        List structFields = new ArrayList();
        structFields.add(DataTypes.createStructField("label",DataTypes.DoubleType,true));
        structFields.add(DataTypes.createStructField("feture1",DataTypes.DoubleType,true));
        structFields.add(DataTypes.createStructField("feture2",DataTypes.DoubleType,true));
        structFields.add(DataTypes.createStructField("feture3",DataTypes.DoubleType,true));

        StructType structType = DataTypes.createStructType(structFields);
        Dataset<Row> persistDF = sqlContext.createDataFrame(rowRDD,structType);
        persistDF.write().mode("append").jdbc(url,"bayes_result",connectionProperties);

        jsc.close();
    }
}
