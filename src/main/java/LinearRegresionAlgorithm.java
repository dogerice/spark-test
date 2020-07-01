import com.alibaba.fastjson.JSONObject;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;



import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.*;
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
 * @date:2019/5/27
 * @des 线性回归
 */
public class LinearRegresionAlgorithm {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("LinearRegresionAlgorithm");
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

        System.out.println(testTable.toJSONString());
        System.out.println(trainTable.toJSONString());
        System.out.println(resultTable.toJSONString());
        System.out.println(operationalParam.toJSONString());

        //训练表信息
/*        String trainTableUrl="jdbc:mysql://"+trainTable.getString("database_address")+":"+trainTable.getString("port")+"/"+
                trainTable.getString("database")+"?useUnicode=true&characterEncoding=utf-8&useSSL=false";*/
        String trainTableUrl = "jdbc:oracle:thin:@"+trainTable.getString("database_address")+":"+
                trainTable.getString("port")+":"+trainTable.getString("database");
        String trainTableName=trainTable.getString("tablename");
        Properties trainConnProp = new Properties();
        trainConnProp.put("user",trainTable.getString("account"));
        trainConnProp.put("password",trainTable.getString("password"));
//        trainConnProp.put("driver","com.mysql.jdbc.Driver");
        trainConnProp.put("driver","oracle.jdbc.driver.OracleDriver");
        String trainLabelCol = operationalParam.getString("train_label_col");
        String[] trainFetureCol = operationalParam.getString("train_feture_col").split(" ");
        int trainFetureNum = trainFetureCol.length;

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
        String[] testFetureCol = operationalParam.getString("test_feture_col").split(" ");
        int testFetureNum = testFetureCol.length;

        //结果表信息

/*        String resultTableUrl="jdbc:mysql://"+resultTable.getString("database_address")+":"+resultTable.getString("port")+"/"+
                resultTable.getString("database")+"?useUnicode=true&characterEncoding=utf-8&useSSL=false";*/
        String resultTableUrl = "jdbc:oracle:thin:@"+resultTable.getString("database_address")+":"+
                resultTable.getString("port")+":"+resultTable.getString("database");
        String resultTableName=resultTable.getString("tablename");
        Properties resultConnProp = new Properties();
        resultConnProp.put("user",resultTable.getString("account"));
        resultConnProp.put("password",resultTable.getString("password"));
        resultConnProp.put("driver","oracle.jdbc.driver.OracleDriver");
        String resultLabelCol = operationalParam.getString("result_label_col");
        String[] resultFetureCol = operationalParam.getString("result_feture_col").split(" ");
        int resultFetureNum = resultFetureCol.length;

        //算法执行参数
        double stepSize = Double.parseDouble(operationalParam.get("stepSize").toString());
        int iterateNum = Integer.parseInt(operationalParam.get("iterateNum").toString());

        Dataset<Row> trainRows = sqlContext.read().jdbc(trainTableUrl,trainTableName,trainConnProp).select(trainLabelCol,trainFetureCol);
        trainRows.show();
//        System.out.println("dataset test col"+trainRows.javaRDD());

        JavaRDD<LabeledPoint> trainRDD = trainRows.javaRDD().map(new Function<Row, LabeledPoint>() {
            @Override
            public LabeledPoint call(Row row) throws Exception {

                double[] feture_doubles = new double[trainFetureNum];
                double labelDouble = Double.parseDouble(row.get(0).toString());
                for (int i = 0; i < trainFetureNum; i++) {
                    feture_doubles[i] = Double.parseDouble(row.get(i + 1).toString());
                }
                return new LabeledPoint(labelDouble, Vectors.dense(feture_doubles));
            }
        });
        trainRDD.cache();

//        int numIterations = 100;
//        double stepSize = 0.0001;
        LinearRegressionModel model1 = LinearRegressionWithSGD.train(JavaRDD.toRDD(trainRDD), iterateNum,stepSize);
//        RidgeRegressionModel model2 = RidgeRegressionWithSGD.train(JavaRDD.toRDD(trainRDD), numIterations);
//        LassoModel model3 = LassoWithSGD.train(JavaRDD.toRDD(trainRDD), numIterations);

        JavaPairRDD<Double, Double> valuesAndPreds1 = trainRDD.mapToPair(point ->
                new Tuple2<>(model1.predict(point.features()), point.label()));
        double MSE1 = valuesAndPreds1.mapToDouble(pair -> {
            double diff = pair._1() - pair._2();
            return diff * diff;
        }).mean();

        /*JavaPairRDD<Double, Double> valuesAndPreds2 = trainRDD.mapToPair(point ->
                new Tuple2<>(model2.predict(point.features()), point.label()));
        double MSE2 = valuesAndPreds2.mapToDouble(pair -> {
            double diff = pair._1() - pair._2();
            return diff * diff;
        }).mean();

        JavaPairRDD<Double, Double> valuesAndPreds3 = trainRDD.mapToPair(point ->
                new Tuple2<>(model3.predict(point.features()), point.label()));
        double MSE3 = valuesAndPreds3.mapToDouble(pair -> {
            double diff = pair._1() - pair._2();
            return diff * diff;
        }).mean();*/



//        LinearRegressionModel model1 = LinearRegressionWithSGD.train(trainRDD.rdd(),100,0.1);
//        RidgeRegressionModel model2 = RidgeRegressionWithSGD.train(trainRDD.rdd(),100);
//        LassoModel model3 = LassoWithSGD.train(trainRDD.rdd(),100);

        Dataset<Row> testRows = sqlContext.read().jdbc(testTableUrl,testTableName,testConnProp).select("id",testFetureCol);
        testRows.show();

        double[] preNum = new double[]{16.0};
        Vector v = Vectors.dense(preNum);
        System.out.println("------------------trainRDD:"+trainRDD.collect());
        System.out.println("------------------valuesAndPreds1:"+valuesAndPreds1.collect());
        System.out.println("------------------MSE1:"+MSE1);
/*        System.out.println("------------------valuesAndPreds2:"+valuesAndPreds2.collect());
        System.out.println("------------------MSE2:"+MSE2);
        System.out.println("------------------valuesAndPreds3:"+valuesAndPreds3.collect());
        System.out.println("------------------MSE3:"+MSE3);*/
//        System.out.println("predict1"+model1.predict(v));
//        System.out.println("predict2"+model1.predict(Vectors.dense(2.5)));
//        System.out.println("predict3"+model1.predict(Vectors.dense(-0.5)));
//        System.out.println("predict2"+model2.predict(v));
//        System.out.println("predict3"+model3.predict(v));

        JavaRDD<Row> resRowRDD = testRows.javaRDD().map(new Function<Row, Row>() {
            @Override
            public Row call(Row row) throws Exception {

                double[] testFetureDoubles = new double[testFetureNum];
                int id = Integer.parseInt(row.get(0).toString());
                for (int i = 0; i < testFetureNum; i++) {
                    //第一列为无用列(id) 从第二列开始取
                    testFetureDoubles[i] = Double.parseDouble(row.get(i + 1).toString());
                }
                Vector testFetureVector = Vectors.dense(testFetureDoubles);
                double resLabel = model1.predict(testFetureVector);
                double[] resLabelArr = {resLabel};
                double[] idArr = {id};
                double[] newArr1 = ArrayUtils.addAll(idArr, resLabelArr);
                double[] newArr2 = ArrayUtils.addAll(newArr1, testFetureDoubles);

                Object[] newRowObjectArr = new Object[newArr2.length];
                newRowObjectArr[0] = id;
                for (int i = 1; i < newRowObjectArr.length; i++) {

                    newRowObjectArr[i] = newArr2[i];
                }
                return RowFactory.create(newRowObjectArr);
            }
        });
        System.out.println("------------------------resRowRDD:  "+resRowRDD.collect());
        List structFields = new ArrayList();
        structFields.add(DataTypes.createStructField("id",DataTypes.IntegerType,true));
        structFields.add(DataTypes.createStructField(resultLabelCol,DataTypes.DoubleType,true));
        for (String s:resultFetureCol){
            structFields.add(DataTypes.createStructField(s,DataTypes.DoubleType,true));
        }
        StructType structType = DataTypes.createStructType(structFields);
        Dataset<Row> resDF = sqlContext.createDataFrame(resRowRDD,structType);
        resDF.show();
        resDF.write().mode("append").jdbc(resultTableUrl,resultTableName,resultConnProp);
        jsc.close();

    }
}
