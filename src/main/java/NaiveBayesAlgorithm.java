import com.alibaba.fastjson.JSONObject;
import org.apache.commons.lang3.ArrayUtils;
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
import java.util.Base64;
import java.util.List;
import java.util.Properties;


/**
 * @Auther:liaohongbing@hisense.com
 * @date:2019/5/1
 * @des
 */
public class NaiveBayesAlgorithm {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("NaiveBayesAlgorithm");
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
//        resultConnProp.put("driver","com.mysql.jdbc.Driver");
        resultConnProp.put("driver","oracle.jdbc.driver.OracleDriver");
        String resultLabelCol = operationalParam.getString("result_label_col");
        String[] resultFetureCol = operationalParam.getString("result_feture_col").split(" ");
        int resultFetureNum = resultFetureCol.length;

/*
        String url = "jdbc:mysql://10.16.4.67:3306/spark_test?useUnicode=true&characterEncoding=utf-8&useSSL=false";
        System.out.println(url);
        //查找的表名
        String table = "bayes_train_data";
        //增加数据库的用户名(user)密码(password),指定test数据库的驱动(driver)
        Properties connectionProperties = new Properties();
        connectionProperties.put("user","root");
        connectionProperties.put("password","root");
        connectionProperties.put("driver","com.mysql.jdbc.Driver");*/

        //SparkJdbc读取表内容
        // 读取表数据
//        String trainLabelCol = "label";
//        String trainFetureCol[] = {"feture1","feture2","feture3"};
//        int trainFetureNum = 3;

//        String testFetureCol[] = {"feture1","feture2","feture3"};
//        int testFetureNum = 3;

        Dataset<Row> trainRows = sqlContext.read().jdbc(trainTableUrl,trainTableName,trainConnProp).select(trainLabelCol,trainFetureCol);
        trainRows.show();
        JavaRDD<LabeledPoint> trainRDD = trainRows.javaRDD().map(new Function<Row, LabeledPoint>() {
            @Override
            public LabeledPoint call(Row row) throws Exception {
                /*String[] strs = row.toString().split(",|\\[|\\]");
                Double[] dbs = new Double[strs.length];
                for (int i = 0; i<strs.length;i++){
                    dbs[i]=Double.parseDouble(strs[i]);
                }
                List<List<Double>> lists = splitArr(Arrays.asList(dbs), 0);
                double labelDouble = lists.get(0).get(0);
                Double[] fetureDouble = lists.get(1).toArray(new Double[0]);
                double[] feture_double = Arrays.stream(fetureDouble).mapToDouble(Double::valueOf).toArray();*/

                double[] feture_doubles = new double[trainFetureNum];
                double labelDouble = new Double(row.get(0).toString());
                for(int i = 0;i<trainFetureNum;i++){
                    feture_doubles[i] = Double.parseDouble(row.get(i+1).toString());
                }
                return new LabeledPoint(labelDouble,Vectors.dense(feture_doubles));
            }
        });
        System.out.println("————————trainRDD"+trainRDD.collect());

        NaiveBayesModel model = NaiveBayes.train(trainRDD.rdd());

        //select第一个参数没用 只是为了用select（String,String[]）动态取列 测试集取参时从第二列开始取
        Dataset<Row> testRows = sqlContext.read().jdbc(testTableUrl,testTableName,testConnProp).select("id",testFetureCol);
        testRows.show();

        JavaRDD<Row> resRowRDD = testRows.javaRDD().map(new Function<Row, Row>() {
            @Override
            public Row call(Row row) throws Exception {

                double[] testFetureDoubles = new double[testFetureNum];
                int id = Integer.parseInt(row.get(0).toString());
                for (int i = 0; i < testFetureNum; i++) {
                    //第一列为无用列 从第二列开始取
                    testFetureDoubles[i] = Double.parseDouble(row.get(i + 1).toString());
                }
                Vector testFetureVector = Vectors.dense(testFetureDoubles);
                double resLabel = model.predict(testFetureVector);
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

//        System.out.println("————————resRDD:"+resRowRDD.collect());

/*        JavaRDD<Row> resRowRdd = resRDD.map(new Function<LabeledPoint, Row>() {

            @Override
            public Row call(LabeledPoint labeledPoint) throws Exception {
                double[] resLabel = {labeledPoint.label()};
                double[] resFeatures = labeledPoint.features().toArray();
                double[] newRowDoubles =  ArrayUtils.addAll(resLabel,resFeatures);
                //动态构建row需要转为Object[]
                Object[] newRowObjects = new Object[newRowDoubles.length];
                for (int i= 0;i<newRowDoubles.length;i++){
                    newRowObjects[i] = newRowDoubles[i];
                }
//                Object[] test = {1.0,2.0,3.0,4.0};
                return RowFactory.create(newRowObjects);
            }
        });*/

        System.out.println("——————————resRowRDD:"+resRowRDD.collect());

        List structFields = new ArrayList();
        structFields.add(DataTypes.createStructField("id",DataTypes.IntegerType,true));
        structFields.add(DataTypes.createStructField(resultLabelCol,DataTypes.DoubleType,true));
        for (String s:resultFetureCol){
            structFields.add(DataTypes.createStructField(s,DataTypes.DoubleType,true));
        }

/*        structFields.add(DataTypes.createStructField("label",DataTypes.DoubleType,true));
        structFields.add(DataTypes.createStructField("feture1",DataTypes.DoubleType,true));
        structFields.add(DataTypes.createStructField("feture2",DataTypes.DoubleType,true));
        structFields.add(DataTypes.createStructField("feture3",DataTypes.DoubleType,true));*/

        StructType structType = DataTypes.createStructType(structFields);
        Dataset<Row> resDF = sqlContext.createDataFrame(resRowRDD,structType);
        resDF.show();
        resDF.write().mode("append").jdbc(resultTableUrl,resultTableName,resultConnProp);

        jsc.close();
    }
}
