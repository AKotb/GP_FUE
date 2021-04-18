/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.gpfue.servlets;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import javax.servlet.RequestDispatcher;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 *
 * @author akotb
 */
public class HomeServlet extends HttpServlet {

    private String PYTHON_INTERPRETER_PATH = "C:/Users/akotb/AppData/Local/Programs/Python/Python39/python.exe";
    private String PYTHON_SCRIPT_PATH = "C:/Users/akotb/OneDrive/Documents/NetBeansProjects/GP_FUE/web/resources/codeTest.py";

    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        response.setContentType("text/html");
        PrintWriter out = response.getWriter();
        String scene = request.getParameter("scene");
        String year = request.getParameter("year");
        String algorithm = request.getParameter("algorithm");
        String name = request.getParameter("name");
        if (scene.isEmpty() || year.isEmpty() || algorithm.isEmpty() || name.isEmpty()) {
            RequestDispatcher rd = request.getRequestDispatcher("home.jsp");
            out.println("<font color=red>Please fill all the fields</font>");
            rd.include(request, response);
        } else {
            // Creating command that takes the string variables. 
            String command = PYTHON_INTERPRETER_PATH + " " + PYTHON_SCRIPT_PATH + " -s " + scene + " -y \""
                    + year + "\" -g " + algorithm + " -n " + name;

            System.out.println("====================================================");
            System.out.println("Python Command: " + command);
            System.out.println("====================================================");

            // It takes the command and goes to the python code(EX: run_svm)
            ProcessBuilder builder = new ProcessBuilder("cmd.exe", "/c", command);
            builder.redirectErrorStream(true);
            Process p = builder.start();
            
            // Reads the python output code.
            BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line;
            while (true) {
                line = r.readLine();
                if (line == null) {
                    break;
                }
                System.out.println(line);
                request.setAttribute("pythonresult", line);
            }
            System.out.println("After execution of the python script");
            
            // Viewing the output result after finishing the python code.
            RequestDispatcher rd = request.getRequestDispatcher("success.jsp");
            rd.forward(request, response);
        }
    }
}
