import com.alibaba.fastjson.JSONObject;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.util.*;

/**
 * @Auther:liaohongbing@hisense.com
 * @date:2019/7/2
 * @des 决策树
 */
public class DecisionTreeClassification {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("DecisionTreeClassification");
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
        Integer numClasses = Integer.parseInt(operationalParam.get("numClasses").toString());
        Integer maxDepth = Integer.parseInt(operationalParam.get("maxDepth").toString());
        Integer maxBins = Integer.parseInt(operationalParam.get("maxBins").toString());

        Dataset<Row> trainRows = sqlContext.read().jdbc(trainTableUrl,trainTableName,trainConnProp).select(trainLabelCol,trainFetureCol);
        trainRows.show();

        JavaRDD<LabeledPoint> trainRDD = trainRows.javaRDD().map(new Function<Row, LabeledPoint>() {
            @Override
            public LabeledPoint call(Row row) throws Exception {
                //s
                double[] feture_doubles = new double[trainFetureNum];
                double labelDouble = new Double(row.get(0).toString());
                for(int i = 0;i<trainFetureNum;i++){
                    feture_doubles[i] = Double.parseDouble(row.get(i+1).toString());
                }
                return new LabeledPoint(labelDouble, Vectors.dense(feture_doubles));
            }
        });

        //设置决策树参数，训练模型
//        Integer numClasses = 3;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        String impurity = "gini";
//        Integer maxDepth = 5;
//        Integer maxBins = 32;
        final DecisionTreeModel tree_model = DecisionTree.trainClassifier(trainRDD, numClasses,categoricalFeaturesInfo, impurity, maxDepth, maxBins);
        System.out.println("决策树模型：");
        System.out.println(tree_model.toDebugString());

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
                double resLabel = tree_model.predict(testFetureVector);
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

        System.out.println("——————————resRowRDD:"+resRowRDD.collect());

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
