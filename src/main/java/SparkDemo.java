import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
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
import scala.Serializable;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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
            System.out.println("_________________+______________________"+arg);
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
        String trainLabelCol = "label";
        String trainFetureCol[] = {"feture1","feture2","feture3"};
        int trainFetureNum = 3;

        String testFetureCol[] = {"feture1","feture2","feture3"};
        int testFetureNum = 3;

        Dataset<Row> trainRows = sqlContext.read().jdbc(url,table,connectionProperties).select(trainLabelCol,trainFetureCol);
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
        Dataset<Row> testRows = sqlContext.read().jdbc(url,"bayes_test_data",connectionProperties).select(trainFetureCol[0],trainFetureCol);
        testRows.show();

        JavaRDD<LabeledPoint> resRDD = testRows.javaRDD().map(new Function<Row, LabeledPoint>() {
            @Override
            public LabeledPoint call(Row row) throws Exception {

                double[] testFetureDoubles = new double[testFetureNum];
                for (int i = 0;i<testFetureNum;i++){
                    //第一列为无用列 从第二列开始取
                    testFetureDoubles[i] = Double.parseDouble(row.get(i+1).toString());
                }
                Vector testFetureVector = Vectors.dense(testFetureDoubles);
                double resLabel = model.predict(testFetureVector);
                return new LabeledPoint(resLabel,testFetureVector);
            }
        });

        System.out.println("————————resRDD:"+resRDD.collect());

        JavaRDD<Row> resRowRdd = resRDD.map(new Function<LabeledPoint, Row>() {

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
        });

        System.out.println("——————————resRowRDD:"+resRowRdd.collect());

        String resultLabelCol ="label";
        String[] resultFetureCol = {"feture1","feture2","feture3"};

        List structFields = new ArrayList();
        structFields.add(DataTypes.createStructField(resultLabelCol,DataTypes.DoubleType,true));
        for (String s:resultFetureCol){
            structFields.add(DataTypes.createStructField(s,DataTypes.DoubleType,true));
        }

/*        structFields.add(DataTypes.createStructField("label",DataTypes.DoubleType,true));
        structFields.add(DataTypes.createStructField("feture1",DataTypes.DoubleType,true));
        structFields.add(DataTypes.createStructField("feture2",DataTypes.DoubleType,true));
        structFields.add(DataTypes.createStructField("feture3",DataTypes.DoubleType,true));*/

        StructType structType = DataTypes.createStructType(structFields);
        Dataset<Row> resDF = sqlContext.createDataFrame(resRowRdd,structType);
        resDF.show();
        resDF.write().mode("append").jdbc(url,"bayes_result",connectionProperties);

        jsc.close();

    }

    public static <T>  List<List<T>> splitArr( List<T> arr,int index){
        int arrLen = arr.size();
        int arr1Len = index+1;
        int arr2Len = arrLen - arr1Len;

        List<T> newArr1 = new ArrayList<T>();
        List<T> newArr2 = new ArrayList<T>();

        for (int i=0;i<=index;i++){
            newArr1.add(arr.get(i));
        }

        for (int i=index+1;i<arrLen;i++){
            newArr2.add(arr.get(i));
        }
        List<List<T>> result = new ArrayList<List<T>>();
        result.add(newArr1);
        result.add(newArr2);
        return result;
    }

/*    public static void main(String[] args) {

        String[] strs = "123456851".split("");
        Double[] dbs = new Double[strs.length];
        for (int i = 0; i<strs.length;i++){
            dbs[i]=Double.parseDouble(strs[i]);
        }

        List<Double> dbList = Arrays.asList(dbs);
        List<List<Double>> result = splitArr(dbList, 3);

        System.out.println(result.get(0)+"----"+result.get(1));

        Double[] doubles = result.get(0).toArray(new Double[0]);


    }*/

}
