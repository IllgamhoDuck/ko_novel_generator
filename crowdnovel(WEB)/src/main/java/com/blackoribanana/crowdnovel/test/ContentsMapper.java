package com.blackoribanana.crowdnovel.test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public interface ContentsMapper {
  public void insContentsData(Map<String, String> param);
  public List<HashMap<String, Object>> getContentsList();
  public int getContentsCount();
  public int getContentsMaxId();
}