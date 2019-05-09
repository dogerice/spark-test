import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;

import java.util.Properties;

/**
 * @Auther:liaohongbing@hisense.com
 * @date:2019/5/9
 * @des
 */
public class SparkDemo {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("SparkDemo");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        for(String arg :args){
            System.out.println("_______________________________________"+arg);
        }
        SQLContext sqlContext = new SQLContext(jsc);

        String url = "jdbc:mysql://10.16.4.67:3306/spark_test?useUnicode=true&characterEncoding=utf-8&useSSL=false";
        System.out.println(url);
        //查找的表名
        String table = "bayes_train_data";
        //增加数据库的用户名(user)密码(password),指定test数据库的驱动(driver)
        Properties connectionProperties = new Properties();
        connectionProperties.put("user","root");
        connectionProperties.put("password","root");
        connectionProperties.put("driver","com.mysql.jdbc.Driver");

        //SparkJdbc读取表内容
        // 读取表数据
        String[] cols = {"label","feture1","feture2","feture3"};


        Dataset<Row> rows = sqlContext.read().jdbc(url,table,connectionProperties).select();

        rows.show();

        jsc.close();

    }
}
