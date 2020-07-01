import com.alibaba.fastjson.JSONObject;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.fpm.AssociationRules;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowthModel;
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
 * @des 关联分析
 */
public class AssociationAnalysis {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("DecisionTreeClassification");
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
        String trainAssociationCol = operationalParam.getString("train_association_col");

        //结果表信息
        String resultTableUrl = "jdbc:oracle:thin:@"+resultTable.getString("database_address")+":"+
                resultTable.getString("port")+":"+resultTable.getString("database");
        String resultTableName=resultTable.getString("tablename");
        Properties resultConnProp = new Properties();
        resultConnProp.put("user",resultTable.getString("account"));
        resultConnProp.put("password",resultTable.getString("password"));
        resultConnProp.put("driver","oracle.jdbc.driver.OracleDriver");
        String resultRuleCol = operationalParam.getString("result_rule_col");
        String resultConfCol = operationalParam.getString("result_conf_col");

        //运行参数
        double minSupport = Double.parseDouble(operationalParam.get("min_support").toString());//最小支持度
        double minConfidence = Double.parseDouble(operationalParam.get("min_confidence").toString());//最小置信度


        Dataset<Row> trainRows = sqlContext.read().jdbc(trainTableUrl,trainTableName,trainConnProp).select(trainAssociationCol);
        trainRows.show();

        JavaRDD<List<String>> trainRDD = trainRows.javaRDD().map(new Function<Row, List<String>>() {
            @Override
            public List<String> call(Row row) throws Exception {
                String association = row.getString(0);
                List<String> itemsArray = Arrays.asList(association.split(" "));

                return itemsArray;
            }
        });

        System.out.println("_______________trainRDD:   "+trainRDD.collect());

        FPGrowth fpg = new FPGrowth().setMinSupport(minSupport).setNumPartitions(10);

        FPGrowthModel<String> model = fpg.run(trainRDD);

        for (FPGrowth.FreqItemset<String> itemset: model.freqItemsets().toJavaRDD().collect()) {
            System.out.println("[" + itemset.javaItems() + "], " + itemset.freq());
        }

//        double minConfidence = 0.0;  //最小置信度
/*        for (AssociationRules.Rule<String> rule : model.generateAssociationRules(minConfidence).toJavaRDD().collect()) {
            System.out.println(
                    rule.javaAntecedent() + " => " + rule.javaConsequent() + ", " + rule.confidence());
        }*/

        JavaRDD<Row> resRowRDD = model.generateAssociationRules(minConfidence).toJavaRDD().map(new Function<AssociationRules.Rule<String>, Row>() {

            @Override
            public Row call(AssociationRules.Rule<String> rule) throws Exception {

                double confi = rule.confidence();
                String ruleString = rule.javaAntecedent() + " => " + rule.javaConsequent();
                Object[] newRowObjectArr = new Object[2];
                newRowObjectArr[0] = ruleString;
                newRowObjectArr[1] = confi;

                return RowFactory.create(newRowObjectArr);
            }
        });

        System.out.println("——————————resRowRDD:"+resRowRDD.collect());

        List structFields = new ArrayList();
        structFields.add(DataTypes.createStructField(resultRuleCol,DataTypes.StringType,true));
        structFields.add(DataTypes.createStructField(resultConfCol,DataTypes.DoubleType,true));

        StructType structType = DataTypes.createStructType(structFields);
        Dataset<Row> resDF = sqlContext.createDataFrame(resRowRDD,structType);
        resDF.show();

        resDF.write().mode("append").jdbc(resultTableUrl,resultTableName,resultConnProp);


    }
}
