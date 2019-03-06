package com.blackoribanana.crowdnovel.controller;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.blackoribanana.crowdnovel.service.HttpConnectionUtil;
import com.blackoribanana.crowdnovel.test.ContentsMapper;
import com.blackoribanana.crowdnovel.test.TestMapper;

import org.apache.http.NameValuePair;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class KMController {
    @Autowired
    TestMapper tm;
    @Autowired
    ContentsMapper conMap;

    @GetMapping("/{name}.html")
    public String miso(@PathVariable String name, Model model) {
        System.out.println("호호");
        model.addAttribute("pageName", name);
        model.addAttribute("test", "미소돼지");
        return "page";
    }

    @GetMapping("/getTest")
    public ResponseEntity<String> miso2(Model model) {
        System.out.println("테스ㅡ");
        String a = tm.getTestData();
        return ResponseEntity.ok().body(a);
    }

    @GetMapping("/getContentsList")
    @ResponseBody
    public List<HashMap<String,Object>> miso3(Model model) {
        System.out.println("yyeyeye");
        List<HashMap<String,Object>> contentList = conMap.getContentsList();
        return contentList;
    }

    @PostMapping("/insContents")
    public String miso4(@RequestBody Map<String, Object> payload, Model model) {
        System.out.println("checkcheck");
        
        Map<String, String> param = new HashMap<String, String>();
        param.put("USER_NAME", payload.get("userName").toString());
        param.put("CONTENTS_TYPE", "HUMAN");
        param.put("TEXT", payload.get("inputText").toString());
        conMap.insContentsData(param);
        int count = conMap.getContentsCount();
        int mid = conMap.getContentsMaxId();
        System.out.println("갯수는~"+count);
        String isFirst = "N";
        if (count == 1) {
            isFirst = "Y";
        }
        HttpConnectionUtil.connectHttpGet("{api_server}/api/put_Human_txt?contents_id="+mid+"&is_first="+isFirst);
        return "main";
    }
 
}