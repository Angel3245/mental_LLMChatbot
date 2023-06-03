-- MySQL dump 10.13  Distrib 8.0.30, for Win64 (x86_64)
--
-- Host: localhost    Database: reddit_subreddit_anxiety
-- ------------------------------------------------------
-- Server version	8.0.30

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `comment`
--

DROP TABLE IF EXISTS `comment`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `comment` (
  `id` varchar(50) NOT NULL,
  `name` varchar(50) DEFAULT NULL,
  `user_id` varchar(50) DEFAULT NULL,
  `subreddit_id` varchar(50) DEFAULT NULL,
  `body` longtext,
  `body_html` longtext,
  `created_utc` datetime DEFAULT NULL,
  `downs` int DEFAULT NULL,
  `no_follow` tinyint DEFAULT NULL,
  `score` int DEFAULT NULL,
  `send_replies` int DEFAULT NULL,
  `stickied` tinyint DEFAULT NULL,
  `ups` int DEFAULT NULL,
  `permalink` text,
  `parent_id` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `post`
--

DROP TABLE IF EXISTS `post`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `post` (
  `id` varchar(50) NOT NULL,
  `name` varchar(50) DEFAULT NULL,
  `user_id` varchar(50) DEFAULT NULL,
  `subreddit_id` varchar(50) DEFAULT NULL,
  `permalink` text,
  `body` longtext,
  `body_html` longtext,
  `title` text,
  `created_utc` datetime DEFAULT NULL,
  `downs` int DEFAULT NULL,
  `no_follow` tinyint DEFAULT NULL,
  `score` int DEFAULT NULL,
  `send_replies` tinyint DEFAULT NULL,
  `stickied` tinyint DEFAULT NULL,
  `ups` int DEFAULT NULL,
  `link_flair_text` varchar(50) DEFAULT NULL,
  `link_flair_type` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `subreddit`
--

DROP TABLE IF EXISTS `subreddit`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `subreddit` (
  `id` varchar(50) NOT NULL,
  `name` varchar(50) DEFAULT NULL,
  `display_name` varchar(50) DEFAULT NULL,
  `display_name_prefixed` varchar(50) DEFAULT NULL,
  `title` text,
  `url` text,
  `created_utc` datetime DEFAULT NULL,
  `description` text,
  `description_html` text,
  `public_description` text,
  `public_description_html` text,
  `submit_text` text,
  `submit_text_html` text,
  `submit_text_label` varchar(100) DEFAULT NULL,
  `accept_followers` tinyint DEFAULT NULL,
  `accounts_active` int DEFAULT NULL,
  `active_user_count` int DEFAULT NULL,
  `can_assign_link_flair` tinyint DEFAULT NULL,
  `can_assign_user_flair` tinyint DEFAULT NULL,
  `lang` varchar(5) DEFAULT NULL,
  `subscribers` int DEFAULT NULL,
  `wiki_enabled` tinyint DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `user`
--

DROP TABLE IF EXISTS `user`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `user` (
  `id` varchar(50) NOT NULL,
  `name` varchar(50) DEFAULT NULL,
  `created_utc` datetime DEFAULT NULL,
  `subreddit` varchar(50) DEFAULT NULL,
  `awardee_karma` int DEFAULT NULL,
  `awarder_karma` int DEFAULT NULL,
  `comment_karma` int DEFAULT NULL,
  `has_subscribed` tinyint DEFAULT NULL,
  `has_verified_email` tinyint DEFAULT NULL,
  `hide_from_robots` tinyint DEFAULT NULL,
  `icon_img` text,
  `is_employee` tinyint DEFAULT NULL,
  `is_gold` tinyint DEFAULT NULL,
  `is_mod` tinyint DEFAULT NULL,
  `link_karma` int DEFAULT NULL,
  `pref_show_snoovatar` tinyint DEFAULT NULL,
  `snoovatar_img` text,
  `total_karma` int DEFAULT NULL,
  `verified` tinyint DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2023-02-16 10:46:07
