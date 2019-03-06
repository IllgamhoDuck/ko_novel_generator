package com.blackoribanana.crowdnovel.controller;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;

@Controller
public class TestController {
 
    @GetMapping("")
    public String page(@PathVariable String name, Model model) {
        System.out.println("하하");
        model.addAttribute("pageName", name);
        return "page";
    }
 
}