import com.alibaba.fastjson.JSONObject;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Properties;
/**
 * @Auther:liaohongbing@hisense.com
 * @date:2019/7/8
 * @des 协同过滤
 */
public class CollaborativeFiltering {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("CollaborativeFiltering");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        SQLContext sqlContext = new SQLContext(jsc);

        System.out.println("get Arg encode "+args[0]);
        System.out.println("get Arg decode "+ new String(Base64.getDecoder().decode(args[0])));
        JSONObject arg = JSONObject.parseObject(new String(Base64.getDecoder().decode(args[0])));
        JSONObject datasource = arg.getJSONObject("datasource");
        JSONObject trainTable = datasource.getJSONObject("train_table");
        JSONObject testTable = datasource.getJSONObject("test_table");
        JSONObject output = arg.getJSONObject("output");
        JSONObject resultTable = output.getJSONObject("result_table");
        JSONObject operationalParam = arg.getJSONObject("operational_param");

        System.out.println(trainTable.toJSONString());
        System.out.println(testTable.toJSONString());
        System.out.println(resultTable.toJSONString());
        System.out.println(operationalParam.toJSONString());

        //训练表（评分）
        String trainTableUrl = "jdbc:oracle:thin:@"+trainTable.getString("database_address")+":"+
                trainTable.getString("port")+":"+trainTable.getString("database");
        String trainTableName=trainTable.getString("tablename");
        Properties trainConnProp = new Properties();
        trainConnProp.put("user",trainTable.getString("account"));
        trainConnProp.put("password",trainTable.getString("password"));
//        trainConnProp.put("driver","com.mysql.jdbc.Driver");
        trainConnProp.put("driver","oracle.jdbc.driver.OracleDriver");
        String trainUserCol = operationalParam.getString("train_user_col");
        String trainItemCol = operationalParam.getString("train_item_col");
        String trainRateCol = operationalParam.getString("train_rate_col");

        //测试表信息
/*        String testTableUrl="jdbc:mysql://"+testTable.getString("database_address")+":"+testTable.getString("port")+"/"+
                testTable.getString("database")+"?useUnicode=true&characterEncoding=utf-8&useSSL=false";*/
        String testTableUrl = "jdbc:oracle:thin:@"+testTable.getString("database_address")+":"+
                testTable.getString("port")+":"+testTable.getString("database");
        String testTableName=testTable.getString("tablename");
        Properties testConnProp = new Properties();
        testConnProp.put("user",testTable.getString("account"));
        testConnProp.put("password",testTable.getString("password"));
//        testConnProp.put("driver","com.mysql.jdbc.Driver");
        testConnProp.put("driver","oracle.jdbc.driver.OracleDriver");
        String testUserCol = operationalParam.getString("test_user_col");
        String testItemCol = operationalParam.getString("test_item_col");


        //结果表信息

        String resultTableUrl = "jdbc:oracle:thin:@"+resultTable.getString("database_address")+":"+
                resultTable.getString("port")+":"+resultTable.getString("database");
        String resultTableName=resultTable.getString("tablename");
        Properties resultConnProp = new Properties();
        resultConnProp.put("user",resultTable.getString("account"));
        resultConnProp.put("password",resultTable.getString("password"));
        resultConnProp.put("driver","oracle.jdbc.driver.OracleDriver");
        String resultUserCol = operationalParam.getString("result_user_col");
        String resultItemCol = operationalParam.getString("result_item_col");
        String resultRateCol = operationalParam.getString("result_rate_col");

        //运行参数
        int rank = Integer.parseInt(operationalParam.get("rank").toString());
        int numIterations = Integer.parseInt((operationalParam.get("numIterations").toString()));


        Dataset<Row> trainRows = sqlContext.read().jdbc(trainTableUrl,trainTableName,trainConnProp).select(trainUserCol,trainItemCol,trainRateCol);
        trainRows.show();


        JavaRDD<Rating> ratingJavaRDD = trainRows.javaRDD().map(new Function<Row, Rating>() {
            @Override
            public Rating call(Row row) throws Exception {
                return new Rating(Integer.parseInt(row.get(0).toString()), Integer.parseInt(row.get(1).toString()), Double.parseDouble(row.get(2).toString()));
            }
        });
        ratingJavaRDD.cache();

        MatrixFactorizationModel model = ALS.train(ratingJavaRDD.rdd(), rank, numIterations, 0.01);
//        System.out.println("pred_______________________________________"+model.predict(1,1));
//        System.out.println("pred_______________________________________"+model.predict(1,2));
//        System.out.println("pred_______________________________________"+model.predict(1,3));

        JavaRDD<Tuple2<Object, Object>> userProducts = ratingJavaRDD.map(r -> new Tuple2<>(r.user(), r.product()));
        System.out.println("__________userProducts: "+userProducts.collect());


        JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
                model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD()
                        .map(r -> new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating()))
        );
        System.out.println("__________predictions: "+predictions.collect());


        JavaRDD<Tuple2<Double, Double>> ratesAndPreds = JavaPairRDD.fromJavaRDD(
                ratingJavaRDD.map(r -> new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating())))
                .join(predictions).values();
        System.out.println("__________ratesAndPreds: "+ratesAndPreds.collect());


        //读取待测数据并预测

        Dataset<Row> testRows = sqlContext.read().jdbc(testTableUrl,testTableName,testConnProp).select(testUserCol,testItemCol);
        testRows.show();

//        JavaRDD<Row> resRowRdd = testRows.javaRDD().map(new Function<Row, Row>() {
//            @Override
//            public Row call(Row row) throws Exception {
//
//                int user = Integer.parseInt(row.get(0).toString());
//                int item = Integer.parseInt(row.get(1).toString());
//                double predRate = model.predict(user, item);
//
//                Object[] resultArr = {user, item, predRate};
//                return RowFactory.create(resultArr);
//            }
//        });

//        System.out.println("------------------------resRowRDD:  "+resRowRdd.collect());

        JavaRDD<Tuple2<Object, Object>> userItems = testRows.javaRDD().map(new Function<Row, Tuple2<Object, Object>>() {
            @Override
            public Tuple2<Object, Object> call(Row row) throws Exception {
                int user = Integer.parseInt(row.get(0).toString());
                int item = Integer.parseInt(row.get(1).toString());
                return new Tuple2<>(user, item);
            }
        });

        JavaPairRDD<Tuple2<Integer, Integer>, Double> resPairRdd = JavaPairRDD.fromJavaRDD(
                model.predict(JavaRDD.toRDD(userItems)).toJavaRDD()
                        .map(r -> new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating()))
        );
        System.out.println("-------------------------resPairRdd: "+resPairRdd.collect());

        JavaRDD<Row> resRowRdd = resPairRdd.map(new Function<Tuple2<Tuple2<Integer, Integer>, Double>, Row>() {
            @Override
            public Row call(Tuple2<Tuple2<Integer, Integer>, Double> resPair) throws Exception {
                int user = resPair._1._1;
                int item = resPair._1._2;
                double predRate = resPair._2;
                return RowFactory.create(user, item, predRate);
            }
        });
        System.out.println("-------------------------------resRowRdd"+resRowRdd.collect());

        List structFields = new ArrayList();
        structFields.add(DataTypes.createStructField(resultUserCol,DataTypes.IntegerType,true));
        structFields.add(DataTypes.createStructField(resultItemCol,DataTypes.IntegerType,true));
        structFields.add(DataTypes.createStructField(resultRateCol,DataTypes.DoubleType,true));
        StructType structType = DataTypes.createStructType(structFields);
        Dataset<Row> resDF = sqlContext.createDataFrame(resRowRdd,structType);
        resDF.show();

        resDF.write().mode("append").jdbc(resultTableUrl,resultTableName,resultConnProp);
        jsc.close();

    }
}
