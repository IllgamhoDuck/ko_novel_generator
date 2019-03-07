package com.blackoribanana.crowdnovel.test;

import java.util.HashMap;
import java.util.List;

public interface TestMapper {
  public String getTestData();
  public List<HashMap<String, String>> getTestList();
  public void insTestData(String a);
}