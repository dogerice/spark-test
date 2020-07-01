import com.alibaba.fastjson.JSONObject;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;

import java.util.Base64;
import java.util.Properties;

/**
 * @Auther:liaohongbing@hisense.com
 * @date:2019/8/8
 * @des
 */
public class Test {

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
//        String trainLabelCol = operationalParam.getString("train_label_col");
//        String[] trainFetureCol = operationalParam.getString("train_feture_col").split(" ");
//        int trainFetureNum = trainFetureCol.length;


        //输出结果
        String resultTableUrl = "jdbc:oracle:thin:@"+resultTable.getString("database_address")+":"+
                resultTable.getString("port")+":"+resultTable.getString("database");
        String resultTableName=resultTable.getString("tablename");
        Properties resultConnProp = new Properties();
        resultConnProp.put("user",resultTable.getString("account"));
        resultConnProp.put("password",resultTable.getString("password"));
//        resultConnProp.put("driver","com.mysql.jdbc.Driver");
        resultConnProp.put("driver","oracle.jdbc.driver.OracleDriver");
//        String resultLabelCol = operationalParam.getString("result_label_col");
//        String[] resultFetureCol = operationalParam.getString("result_feture_col").split(" ");
//        int resultFetureNum = resultFetureCol.length;

        //算法执行参数
//        int clusterNum = Integer.parseInt(operationalParam.get("clusterNum").toString());
//        int iterateNum = Integer.parseInt(operationalParam.get("iterateNum").toString());

//        String dataPath = SparkKMeansCluster.class.getClassLoader().getResource("kmeans_data.txt").getFile();

        Dataset<Row> trainRows = sqlContext.read().jdbc(trainTableUrl,trainTableName,trainConnProp).select("*");
        trainRows.show();


        System.out.println("-------------------------结束------------------------");
        jsc.close();
    }
}
