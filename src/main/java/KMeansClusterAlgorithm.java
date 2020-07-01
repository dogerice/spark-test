import com.alibaba.fastjson.JSONObject;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Properties;

/**
 * @Auther:liaohongbing@hisense.com
 * @date:2019/5/1
 * @des K均值聚类
 */
public class KMeansClusterAlgorithm {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("NaiveBayesAlgorithm");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        SQLContext sqlContext = new SQLContext(jsc);

        System.out.println("get Arg encode "+args[0]);
        System.out.println("get Arg decode "+ new String(Base64.getDecoder().decode(args[0])));
        JSONObject arg = JSONObject.parseObject(new String(Base64.getDecoder().decode(args[0])));
        JSONObject datasource = arg.getJSONObject("datasource");
        JSONObject trainTable = datasource.getJSONObject("train_table");
        JSONObject output = arg.getJSONObject("output");
        JSONObject resultTable = output.getJSONObject("result_table");
        JSONObject operationalParam = arg.getJSONObject("operational_param");

        System.out.println(trainTable.toJSONString());
        System.out.println(resultTable.toJSONString());
        System.out.println(operationalParam.toJSONString());

        //要聚类的数据
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


        //输出结果
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

        //算法执行参数
        int clusterNum = Integer.parseInt(operationalParam.get("clusterNum").toString());
        int iterateNum = Integer.parseInt(operationalParam.get("iterateNum").toString());

//        String dataPath = SparkKMeansCluster.class.getClassLoader().getResource("kmeans_data.txt").getFile();

        Dataset<Row> trainRows = sqlContext.read().jdbc(trainTableUrl,trainTableName,trainConnProp).select("id",trainFetureCol);
        trainRows.show();

        JavaRDD<Vector> trainVectorRDD = trainRows.javaRDD().map(new Function<Row, Vector>() {
            @Override
            public Vector call(Row row) throws Exception {
                double[] featureDoubles = new double[trainFetureNum];
                for (int i=0;i<trainFetureNum;i++){
                    //第一列为id 从第二列开始取特征
                    featureDoubles[i] = Double.parseDouble(row.get(i+1).toString());
                }
                return Vectors.dense(featureDoubles);

            }
        });

        System.out.println("-------------trainVectorRDD"+trainVectorRDD.collect());

        KMeansModel kMeansModel = KMeans.train(trainVectorRDD.rdd(),clusterNum,iterateNum);

        JavaRDD<String> resString = trainVectorRDD.map(new Function<Vector, String>() {
            public String call(Vector v1) throws Exception {
                return v1.toString() + "==>" + kMeansModel.predict(v1);
            }
        });

        System.out.println("---------------------resStringRDD"+resString.collect());


        JavaRDD<Row> resRowRDD = trainRows.javaRDD().map(new Function<Row, Row>() {
            @Override
            public Row call(Row row) throws Exception {
                double[] featureDoubles = new double[trainFetureNum];
                for (int i = 0; i < trainFetureNum; i++) {
                    //第一列为id 从第二列开始取特征
                    featureDoubles[i] = Double.parseDouble(row.get(i + 1).toString());
                }
                Vector v = Vectors.dense(featureDoubles);
                double type = kMeansModel.predict(v);
                int rowId = Integer.parseInt(row.get(0).toString());

                Object[] newRowObjArr = new Object[trainFetureNum + 2];//特征列+id+type
                newRowObjArr[0] = rowId;
                newRowObjArr[1] = type;
                for (int i = 2; i < trainFetureNum + 2; i++) {
                    newRowObjArr[i] = featureDoubles[i - 2];
                }
                return RowFactory.create(newRowObjArr);

            }
        });
        System.out.println("---------------------resRowRDD"+resString.collect());


        List structFields = new ArrayList();
        structFields.add(DataTypes.createStructField("id",DataTypes.IntegerType,true));
        structFields.add(DataTypes.createStructField(resultLabelCol,DataTypes.DoubleType,true));
        for (String s:resultFetureCol){
            structFields.add(DataTypes.createStructField(s,DataTypes.DoubleType,true));
        }


        StructType structType = DataTypes.createStructType(structFields);
        Dataset<Row> resDF = sqlContext.createDataFrame(resRowRDD,structType);
        resDF.show();
        resDF.write().mode("overwrite").jdbc(resultTableUrl,resultTableName,resultConnProp);
        jsc.close();
    }
}
