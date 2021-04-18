<%-- 
    Document   : success
    Created on : Apr 18, 2021, 1:35:09 PM
    Author     : akotb
--%>

<%@page contentType="text/html" pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">
        <title>Parameters Passed</title>
    </head>
    <style>
        table#nat{
            width: 50%;
            background-color: #c48ec5;
        }
    </style>
</head>
<body>  
    <% String scene = request.getParameter("scene");
        String year = request.getParameter("year");
        String algorithm = request.getParameter("algorithm");
        String name = request.getParameter("name");
        String pythonresult = request.getAttribute("pythonresult").toString();%>
    <table id ="nat">
        <tr>
            <td>Scene</td>
            <td><%= scene%></td>
        </tr>
        <tr>
            <td>Year</td>
            <td><%= year%></td>
        </tr>
        <tr>
            <td>Algorithm</td>
            <td><%= algorithm%></td>
        </tr>
        <tr>
            <td>Name</td>
            <td><%= name%></td>
        </tr>
        <tr>
            <td>Python Call Result</td>
            <td><%= pythonresult%></td>
        </tr>
    </table>
</body>
</html>
