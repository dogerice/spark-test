import com.alibaba.fastjson.JSONObject;

import java.util.Base64;

/**
 * @Auther:liaohongbing@hisense.com
 * @date:2019/5/10
 * @des
 */
public class ArgsTest {
    public static void main(String[] args) {
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

    }
}
