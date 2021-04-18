<%-- 
    Document   : Home.jsp
    Created on : Apr 1, 2021, 7:41:30 PM
    Author     : akotb
--%>

<%@page contentType="text/html" pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">
        <title>Test Python Call Form</title>
    </head>
    <style>
        div.ex {
            text-align: right width: 300px;
            padding: 10px;
            border: 5px solid grey;
            margin: 0px
        }
    </style>
    <body>
        <h1>Test Python Call Form</h1>
        <div class="ex">
            <form action="home" method="post">
                <table style="with: 50%">
                    <tr>
                        <td>Enter Scene</td>
                        <td><input type="text" name="scene" /></td>
                    </tr>
                    <tr>
                        <td>Enter Year</td>
                        <td><input type="text" name="year" /></td>
                    </tr>
                    <tr>
                        <td>Enter Algorithm</td>
                        <td><input type="text" name="algorithm" /></td>
                    </tr>
                    <tr>
                        <td>Enter Your Name</td>
                        <td><input type="text" name="name" /></td>
                    </tr>
                </table>
                <input type="submit" value="Submit" />
            </form>
        </div>
    </body>
</html>
