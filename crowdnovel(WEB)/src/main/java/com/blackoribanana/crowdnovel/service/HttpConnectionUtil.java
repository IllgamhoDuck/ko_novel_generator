package com.blackoribanana.crowdnovel.service;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.List;

import org.apache.http.NameValuePair;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.entity.UrlEncodedFormEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class HttpConnectionUtil {
	
	private static final Logger		logger				= LoggerFactory.getLogger(HttpConnectionUtil.class);
	
	
	
	public static String connectHttpPost(String url, List<NameValuePair> params) {
		String resStirng = "";
		CloseableHttpClient httpClient = HttpClients.createDefault();
		HttpPost post = new HttpPost(url);
		post.addHeader("Content-Type", "application/x-www-form-urlencoded");

		try {
			post.setEntity(new UrlEncodedFormEntity(params));
			CloseableHttpResponse execute = httpClient.execute(post);
			logger.info("Response code : " + execute.getStatusLine().getStatusCode());

			BufferedReader reader = new BufferedReader(new InputStreamReader(execute.getEntity().getContent()));
			String inputLine;
			StringBuffer response = new StringBuffer();
			while ((inputLine = reader.readLine()) != null) {
				response.append(inputLine);
			}

			resStirng = response.toString();
			logger.info("Response : " + resStirng);
			reader.close();
			httpClient.close();

		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (ClientProtocolException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return resStirng;
	}

	/**
	 * Http Get Connection Util
	 * 
	 * apiType - discordUser : AuthTokenType, AuthToken (require) - steamUser
	 * 
	 * @param url
	 *            (require)
	 * @param apiType
	 *            (require)
	 * @param AuthTokenType
	 * @param AuthToken
	 * @return
	 */
	public static String connectHttpGet(String url) {
		String resStirng = "";
		CloseableHttpClient httpClient = HttpClients.createDefault();
		HttpGet get = new HttpGet(url);
		get.addHeader("Content-Type", "application/x-www-form-urlencoded");

		try {
			CloseableHttpResponse execute = httpClient.execute(get);
			logger.info("Request url : " + url);
			logger.info("Response code : " + execute.getStatusLine().getStatusCode());

			BufferedReader reader = new BufferedReader(new InputStreamReader(execute.getEntity().getContent()));
			String inputLine;
			StringBuffer response = new StringBuffer();
			while ((inputLine = reader.readLine()) != null) {
				response.append(inputLine);
			}
			resStirng = response.toString();
			logger.info("Response : " + resStirng);
			reader.close();
			httpClient.close();

		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (ClientProtocolException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return resStirng;
	}

}
