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
 * @date:2019/5/27
 * @des
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

        Dataset<Row> trainRows = sqlContext.read().jdbc(trainTableUrl,trainTableName,trainConnProp).select(trainLabelCol,trainFetureCol);
        trainRows.show();



    }
}
